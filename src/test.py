import random
from datetime import datetime

import torch.backends.cudnn as cudnn

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
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)
cudnn.benchmark = True
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with" " --cuda")
# load data
data = util.DATA_LOADER(opt)
print("# of training samples: ", data.ntrain)
logger = util.Logger(log_name)
logger.write("Params : %s \n" % (vars(opt)))
best_gzsl_acc = 0
best_zsl_acc = 0
path = "models/GZSL/" + opt.dataset
netG = model.Generator(opt)
netG.load_state_dict(torch.load(path + "/" + "generator.pt"))
netG.eval()
print(netG)
netE = model.Encoder(opt)
netE.load_state_dict(torch.load(path + "/" + "encoder.pt"))
netE.eval()
print(netE)
if opt.cuda:
    netG.cuda()
    netE.cuda()

syn_feature, syn_label = generate_syn_feature(
    netG, data.unseenclasses, data.attribute, 700#, netF=None, netDec=None
)  # netDec
# Generalized zero-shot learning
if opt.gzsl:
    # Concatenate real seen features with synthesized unseen features
    train_X = torch.cat((data.train_feature, syn_feature), 0)
    train_Y = torch.cat((data.train_label, syn_label), 0)
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
        100,
        opt.syn_num,
        generalized=True,
    )  # netDec

    if best_gzsl_acc < gzsl_cls.H:
        best_acc_seen, best_acc_unseen, best_gzsl_acc = (
            gzsl_cls.acc_seen,
            gzsl_cls.acc_unseen,
            gzsl_cls.H,
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
    100,
    opt.syn_num,
    generalized=False,
)

acc = zsl_cls.acc
if best_zsl_acc < acc:
    best_zsl_acc = acc
    best_zsl_cls = zsl_cls.model.state_dict()
logger.write("ZSL: unseen accuracy=%.4f\n" % (acc))
# reset G to training mode
logger.write("Dataset %s\n" % (opt.dataset))
logger.write("the best ZSL unseen accuracy is %s\n" % (best_zsl_acc))
if opt.gzsl:
    logger.write("the best GZSL seen accuracy is %.4f\n" % (best_acc_seen))
    logger.write("the best GZSL unseen accuracy is %.4f\n" % (best_acc_unseen))
    logger.write("the best GZSL H is %.4f\n" % (best_gzsl_acc))
