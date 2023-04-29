##import
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse

from model import UNet
from dataset import *
from util import *
from matplotlib import pyplot as plt

##parser
parser = argparse.ArgumentParser(description='Train the UNet', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--lr', type=float, default=1e-3, dest='lr')
parser.add_argument('--batch_size', type=int, default=4, dest='batch_size')
parser.add_argument('--num_epoch', type=int, default=100, dest='num_epoch')
parser.add_argument('--num_workers', type=int, default=1, dest='num_workers')

parser.add_argument('--data_dir', type=str, default='./datasets', dest='data_dir')
parser.add_argument('--ckpt_dir', type=str, default='./ckpt', dest='ckpt_dir')
parser.add_argument('--log_dir', type=str, default='./log', dest='log_dir')
parser.add_argument('--result_dir', type=str, default='./results', dest='result_dir')

parser.add_argument('--mode', default= None, type=str, dest='mode')
parser.add_argument('--train_continue', default='off', type=str, dest='train_continue')
parser.add_argument('--device', default='cpu', type=str, dest='device')

args = parser.parse_args()

lr = args.lr
batch_size = args.batch_size
num_epoch = args.num_epoch
num_workers = args.num_workers

data_dir = args.data_dir
ckpt_dir = args.ckpt_dir
log_dir = args.log_dir
result_dir = args.result_dir

mode = args.mode
train_continue = args.train_continue

if args.device == 'cuda':
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        raise ValueError('CUDA is not available')
elif args.device == 'mps':
    if torch.backends.mps.is_available():
        device ='mps'
    else:
        raise ValueError('MPS is not available')
else:
    device = 'cpu'

print('mode : {}'.format(mode))
print('lr : {}, batch_size : {}, num_epoch : {}, num_workers : {}, device : {}, train_continue : {}'.format(lr, batch_size, num_epoch, num_workers, device, train_continue))
##
if not os.path.exists(result_dir):
    os.makedirs(os.path.join(result_dir, 'png'))
    os.makedirs(os.path.join(result_dir, 'npy'))
##
if mode == 'train':
    transform = transforms.Compose([Normalize(), RandomFlip(), ToTensor()])
    dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'), transform=transform)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    dataset_val = Dataset(data_dir=os.path.join(data_dir, 'val'), transform=transform)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)

    num_data_train = len(dataset_train)
    num_data_val = len(dataset_val)

    num_batch_train = int(np.ceil(num_data_train/batch_size))
    num_batch_val = int(np.ceil(num_data_val/batch_size))
else:
    transform = transforms.Compose([Normalize(), ToTensor()])
    dataset_test = Dataset(data_dir=os.path.join(data_dir, 'test'), transform=transform)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    num_data_test = len(dataset_test)

    num_batch_test = int(np.ceil(num_data_test / batch_size))
##
net = UNet().to(device)
fn_loss = nn.BCEWithLogitsLoss().to(device)
#찐 논문대로하려면
fn_loss_real = nn.BCELoss().to(device) #에 정규화텀도 추가해야함.
optim = torch.optim.Adam(net.parameters(), lr=lr)

##
fn_tonumpy = lambda x : x.to('cpu').detach().numpy().transpose(0,2,3,1)
fn_denorm = lambda x, mean, std : (x*std + mean)
fn_class = lambda x : 1.0 * (x > 0.5)
##
writer_train = SummaryWriter(os.path.join(log_dir, 'train'))
writer_val = SummaryWriter(os.path.join(log_dir, 'val'))
##
st_epoch = 0
if train_continue == 'on':
    net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

if mode == 'train': #Training
    for epoch in range(st_epoch + 1, num_epoch + 1):
        net.train()
        loss_arr = []

        for batch, data in enumerate(loader_train, 1):
            label = data['label'].to(device)
            input = data['input'].to(device)
            output = net(input)

            optim.zero_grad()

            loss = fn_loss(output, label)
            loss.backward()

            optim.step()

            loss_arr += [loss.item()]

            print('TRAIN : EPOCH {:4d}/{:4d} | BATCH {:4d}/{:4d} | LOSS {:.4f}'.format(epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr)))

            label = fn_tonumpy(label)
            input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            output = fn_tonumpy(fn_class(output))

            writer_train.add_image('label', label, num_batch_train*(epoch-1) + batch, dataformats='NHWC')
            writer_train.add_image('input', input, num_batch_train*(epoch-1) + batch, dataformats='NHWC')
            writer_train.add_image('output', output, num_batch_train*(epoch-1) + batch, dataformats='NHWC')

        writer_train.add_scalar('loss', np.mean(loss_arr), epoch)

        ##validation

        with torch.no_grad():
            net.train()
            loss_arr = []

            for batch, data in enumerate(loader_val, 1):
                label = data['label'].to(device)
                input = data['input'].to(device)

                output = net(input)

                loss = fn_loss(output, label)
                loss_arr += [loss.item()]

                print('VALID : EPOCH {:4d}/{:4d} | BATCH {:4d}/{:4}'.format(epoch, num_epoch, batch, num_batch_val))

                label = fn_tonumpy(label)
                input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
                output = fn_tonumpy(fn_class(output))

                writer_val.add_image('label', label, num_batch_val*(epoch-1) + batch, dataformats='NHWC')
                writer_val.add_image('input', input, num_batch_val*(epoch-1) + batch, dataformats='NHWC')
                writer_val.add_image('output', output, num_batch_val*(epoch-1) + batch, dataformats='NHWC')

        writer_val.add_scalar('loss', np.mean(loss_arr), epoch)

        if epoch % 25 == 0:
            save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)

    writer_train.close()
    writer_val.close()
else: #Testing
    net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)
    with torch.no_grad():
        net.train()
        loss_arr = []

        for batch, data in enumerate(loader_test, 1):
            label = data['label'].to(device)
            input = data['input'].to(device)

            output = net(input)

            loss = fn_loss(output, label)
            loss_arr += [loss.item()]

            print('TEST : BATCH {:4d}/{:4} | Loss {:.4f}'.format(batch, num_batch_test, np.mean(loss_arr)))

            label = fn_tonumpy(label)
            input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            output = fn_tonumpy(fn_class(output))

        for j in range(label.shape[0]):
            id = num_batch_test * (batch - 1) + j

            plt.imsave(os.path.join(result_dir, 'png', 'label{:4d}.png'.format(id)), label[j].squeeze(), cmap='gray')
            plt.imsave(os.path.join(result_dir, 'png', 'input{:4d}.png'.format(id)), input[j].squeeze(), cmap='gray')
            plt.imsave(os.path.join(result_dir, 'png', 'output{:4d}.png'.format(id)), output[j].squeeze(), cmap='gray')

            np.save(os.path.join(result_dir, 'npy', 'label{:4d}'.format(id)), label[j].squeeze())
            np.save(os.path.join(result_dir, 'npy', 'input{:4d}'.format(id)), input[j].squeeze())
            np.save(os.path.join(result_dir, 'npy', 'output{:4d}'.format(id)), output[j].squeeze())

    print('AVERAGE TEST : BATCH {:4d}/{:4} | Loss {:.4f}'.format(batch, num_batch_test, np.mean(loss_arr)))




