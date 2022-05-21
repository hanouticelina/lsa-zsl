import os
import random
from datetime import datetime

import loguru
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim

import classifiers.classifier_images as classifier
import datasets.image_util as util
import networks.models as model
from utils import *

log_name = (
    "logs/"
    + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    + "_"
    + opt.dataset
    + "_transductive"
)
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
loguru.logger.info("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)
cudnn.benchmark = True
if torch.cuda.is_available() and not opt.cuda:
    loguru.logger.info(
        "WARNING: You have a CUDA device, so you should probably run with" " --cuda"
    )
# load data
data = util.DATA_LOADER(opt)
loguru.logger.info("# of training samples: ", data.ntrain)
logger = util.Logger(log_name)
logger.write("Params : %s \n" % (vars(opt)))
netG = model.Generator(opt)
netD = model.Discriminator_D1(opt)
netD2 = model.Discriminator_D2(opt)
netE = model.Encoder(opt)
cls = model.LinearClassifier(2048, opt.nclass_all)
criterion = nn.CrossEntropyLoss()
loguru.logger.info(netG)
loguru.logger.info(netD)

###########
# Init Tensors
input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
input_att = torch.FloatTensor(
    opt.batch_size, opt.attSize
)  # attSize class-embedding size
input_label = torch.LongTensor(opt.batch_size)
input_res_unpair = torch.FloatTensor(opt.batch_size, opt.resSize)
input_att_unpair = torch.FloatTensor(opt.batch_size, opt.attSize)
input_label_unpair = torch.LongTensor(opt.batch_size)
noise = torch.FloatTensor(opt.batch_size, opt.nz)
noise_mix = torch.FloatTensor(opt.batch_size * 2, opt.nz)
zeros_mix = torch.zeros(opt.nclass_all, opt.nz)
one = torch.tensor(1, dtype=torch.float)
mone = one * -1
input_all_attributes = torch.FloatTensor(opt.nclass_all, opt.attSize)
##########
# Cuda
use_cuda = False
if opt.cuda:
    netD.cuda()
    netG.cuda()
    netD2.cuda()
    netE.cuda()
    cls.cuda()
    use_cuda = True
    input_res = input_res.cuda()
    noise, input_att = noise.cuda(), input_att.cuda()
    input_label = input_label.cuda()
    input_res_unpair = input_res_unpair.cuda()
    input_att_unpair = input_att_unpair.cuda()
    input_label_unpair = input_label_unpair.cuda()
    one = one.cuda()
    mone = mone.cuda()
    noise_mix = noise_mix.cuda()
    zeros_mix = zeros_mix.cuda()
    input_all_attributes = input_all_attributes.cuda()


def sample():
    batch_feature, batch_label, batch_att = data.next_seen_batch(opt.batch_size)
    input_res.copy_(batch_feature)
    input_att.copy_(batch_att)
    input_label.copy_(batch_label)
    batch_feature, batch_label, batch_att = data.next_unseen_batch(opt.batch_size)
    input_res_unpair.copy_(batch_feature)
    input_att_unpair.copy_(batch_att)
    input_label_unpair.copy_(batch_label)


optimizer = optim.Adam(netE.parameters(), lr=opt.lr)
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerD2 = optim.Adam(netD2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerCLS = optim.SGD(cls.parameters(), lr=0.005, momentum=0.9, weight_decay=1e-4)
input_all_attributes.copy_(data.attribute)


best_gzsl_acc = 0
best_zsl_acc = 0
best_generator = None
best_encoder = None
best_gzsl_cls, best_zsl_cls = None, None
for epoch in range(0, opt.nepoch):
    for loop in range(0, 2):
        for i in range(0, data.ntrain, opt.batch_size):
            #########Discriminator training ##############
            for p in netD.parameters():  # unfreeze discrimator
                p.requires_grad = True
            for p in netD2.parameters():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in netG_v update
            # Train D1 and D2
            gp_sum = 0  # lAMBDA VARIABLE
            for iter_d in range(opt.critic_iter):
                sample()
                netD.zero_grad()
                netD2.zero_grad()
                input_resv = Variable(input_res)
                input_attv = Variable(input_att)
                input_res_unpairv = Variable(input_res_unpair)
                input_att_unpairv = Variable(input_att_unpair)
                criticD_real = netD(input_resv, input_attv)
                criticD_real = opt.gammaD * criticD_real.mean()
                criticD_real.backward(mone)
                # non-conditional D on unpaired real data
                criticD_real_v_unpair = netD2(input_res_unpairv)
                criticD_real_v_unpair = opt.gammaD2 * criticD_real_v_unpair.mean()
                if opt.gzsl:  # NO
                    criticD_real_v_unpair_seen = netD2(input_resv)
                    criticD_real_v_unpair += (
                        opt.gammaD2 * criticD_real_v_unpair_seen.mean()
                    )
                criticD_real_v_unpair.backward(mone)

                if opt.encoded_noise:
                    means, log_var = netE(input_resv, input_attv)
                    std = torch.exp(0.5 * log_var)
                    eps = torch.randn([opt.batch_size, opt.latent_size]).cpu()
                    eps = Variable(eps.cuda())
                    z = eps * std + means  # torch.Size([64, 312])
                else:
                    noise.normal_(0, 1)
                    z = Variable(noise)
                fake = netG(z, c=input_attv)

                criticD_fake = netD(fake.detach(), input_attv)
                criticD_fake = opt.gammaD * criticD_fake.mean()
                criticD_fake.backward(one)

                # non-conditional netD_v unpair fake data
                noise.normal_(0, 1)
                z_D2 = Variable(noise)
                fake_v_unpair = netG(z_D2, c=input_att_unpairv)
                criticD_fake_v_unpair = netD2(fake_v_unpair.detach())
                criticD_fake_v_unpair = opt.gammaD2 * criticD_fake_v_unpair.mean()
                if opt.gzsl:
                    criticD_fake_v_unpair_seen = netD2(fake.detach())
                    criticD_fake_v_unpair += (
                        opt.gammaD2 * criticD_fake_v_unpair_seen.mean()
                    )
                criticD_fake_v_unpair.backward(one)

                # gradient penalty
                gradient_penalty = opt.gammaD * calc_gradient_penalty(
                    netD, input_res, fake.data, input_att
                )
                gradient_penalty_v_unpair = opt.gammaD2 * calc_gradient_penalty2(
                    netD2, input_res_unpairv, fake_v_unpair.data
                )
                gradient_penalty_v_unpair.backward()
                # if opt.lambda_mult == 1.1:
                gp_sum += gradient_penalty.data
                gradient_penalty.backward()
                Wasserstein_D = criticD_real - criticD_fake
                D_cost = (
                    criticD_fake - criticD_real + gradient_penalty
                )  # add Y here and #add vae reconstruction loss
                optimizerD.step()
                # non-conditional D, Wasserstein distance
                Wasserstein_D_v2 = criticD_real_v_unpair - criticD_fake_v_unpair
                D_cost_v2 = (
                    criticD_fake_v_unpair
                    - criticD_real_v_unpair
                    + gradient_penalty_v_unpair
                )
                optimizerD2.step()
            gp_sum /= opt.gammaD * opt.lambda1 * opt.critic_iter
            if (gp_sum > 1.05).sum() > 0:
                opt.lambda1 *= 1.1
            elif (gp_sum < 1.001).sum() > 0:
                opt.lambda1 /= 1.1

            #############Generator training ##############
            # Train Generator
            for p in netD.parameters():  # freeze discrimator
                p.requires_grad = False
            for p in netD2.parameters():  # reset requires_grad
                p.requires_grad = False

            cls.zero_grad()
            netE.zero_grad()
            netG.zero_grad()
            input_resv = Variable(input_res)
            input_attv = Variable(input_att)
            input_res_unpairv = Variable(input_res_unpair)
            input_att_unpairv = Variable(input_att_unpair)

            if opt.encoded_noise:
                means, log_var = netE(input_resv, input_attv)
                std = torch.exp(0.5 * log_var)
                eps = torch.randn([opt.batch_size, opt.latent_size]).cpu()
                eps = Variable(eps.cuda())
                z = eps * std + means
                recon_x = netG(z, c=input_attv)
                vae_loss_seen = loss_fn(recon_x, input_resv, means, log_var)
                errG = vae_loss_seen
                criticG_fake = netD(recon_x, input_attv).mean()
                fake = recon_x
            else:
                errG = 0
                noise.normal_(0, 1)
                noisev = Variable(noise)
                fake = netG(noisev, c=input_attv)
                criticG_fake = netD(fake, input_attv).mean()

            G_cost = -criticG_fake
            noise.normal_(0, 1)
            z_D2 = Variable(noise)
            fake_v_unpaired = netG(z_D2, c=input_att_unpairv)
            criticG_fake_v_unpair = netD2(fake_v_unpaired).mean()
            G_cost_v_unpair = -criticG_fake_v_unpair
            errG += opt.gammaG * G_cost + opt.gammaG_D2 * G_cost_v_unpair
            errG.backward()
            if opt.encoded_noise:
                optimizer.step()
            optimizerG.step()
            inputs, targets_a, targets_b, lam = ambiguous_data(
                input_attv,
                input_att_unpair,
                input_label,
                input_label_unpair,
                use_cuda,
            )  # alpha = 1
            inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))

            noise_mix.normal_(0, 1)
            z = Variable(noise_mix)
            outputs = netG(z, c=inputs)
            outputs = cls(outputs)
            loss = ambiguous_criterion(criterion, outputs, targets_a, targets_b, lam)
            loss.backward()
            optimizerCLS.step()
            optimizerG.step()

    logger.write(
        "[%d/%d]  Loss_D: %.4f Loss_G: %.4f, Wasserstein_dist:%.4f,"
        " vae_loss_seen:%.4f\n"
        % (
            epoch,
            opt.nepoch,
            D_cost.data.item(),
            G_cost.data.item(),
            Wasserstein_D.data.item(),
            0,
        )
    )
    netG.eval()
    syn_feature, syn_label = generate_syn_feature(
        netG, data.unseenclasses, data.attribute, 700
    )
    syn_feature_seen, syn_label_seen = generate_syn_feature(
        netG, data.seenclasses, data.attribute, 500
    )
    # Generalized zero-shot learning
    if opt.gzsl:
        # Concatenate real seen features with synthesized unseen features
        perm = torch.randperm(data.train_feature.size(0))
        idx = perm[:400]
        X_real_seen = data.train_feature[idx]
        Y_real_seen = data.train_label[idx]
        X = torch.cat((X_real_seen, syn_feature_seen), 0)
        Y = torch.cat((Y_real_seen, syn_label_seen), 0)
        train_X = torch.cat((X, syn_feature), 0)
        train_Y = torch.cat((Y, syn_label), 0)
        nclass = opt.nclass_all
        # Train GZSL classifier
        gzsl_cls = classifier.CLASSIFIER(
            train_X,
            train_Y,
            data,
            nclass,
            opt.cuda,
            opt.classifier_lr,
            0.5,
            25,
            opt.syn_num,
            generalized=True,
        )

        if best_gzsl_acc < gzsl_cls.H:
            best_acc_seen, best_acc_unseen, best_gzsl_acc = (
                gzsl_cls.acc_seen,
                gzsl_cls.acc_unseen,
                gzsl_cls.H,
            )
            best_generator, best_encoder, best_gzsl_cls = (
                netG.state_dict(),
                netE.state_dict(),
                gzsl_cls.model.state_dict(),
            )
        logger.write(
            "GZSL: seen=%.4f, unseen=%.4f, h=%.4f\n"
            % (gzsl_cls.acc_seen, gzsl_cls.acc_unseen, gzsl_cls.H)
        )
    # Zero-shot learning
    # Train ZSL classifier
    zsl_cls = classifier.CLASSIFIER(
        syn_feature,
        util.map_label(syn_label, data.unseenclasses),
        data,
        data.unseenclasses.size(0),
        opt.cuda,
        opt.classifier_lr,
        0.5,
        25,
        opt.syn_num,
        generalized=False,
    )

    acc = zsl_cls.acc
    if best_zsl_acc < acc:
        best_zsl_acc = acc
        best_zsl_cls = zsl_cls.model.state_dict()
    logger.write("ZSL: unseen accuracy=%.4f\n" % (acc))
    # reset G to training mode
    netG.train()
    netE.train()

#### SAVE MODELS
os.makedirs("models/GZSL/" + opt.dataset, exist_ok=True)
os.makedirs("models/ZSL/" + opt.dataset, exist_ok=True)
torch.save(best_generator, "models/GZSL/" + opt.dataset + "/generator.pt")
torch.save(best_encoder, "models/GZSL/" + opt.dataset + "/encoder.pt")
torch.save(best_gzsl_cls, "models/GZSL/" + opt.dataset + "/gzsl_cls.pt")
torch.save(best_zsl_cls, "models/ZSL/" + opt.dataset + "/zsl_cls.pt")
logger.write("Dataset %s\n" % (opt.dataset))
logger.write("the best ZSL unseen accuracy is %s\n" % (best_zsl_acc))
if opt.gzsl:
    logger.write("the best GZSL seen accuracy is %.4f\n" % (best_acc_seen))
    logger.write("the best GZSL unseen accuracy is %.4f\n" % (best_acc_unseen))
    logger.write("the best GZSL H is %.4f\n" % (best_gzsl_acc))
