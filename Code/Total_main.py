import argparse
import main_shallow
import main_deep
import p

parser = argparse.ArgumentParser(description='BCIC 4-2a')

parser.add_argument('--mode', default="train", choices=['train', 'test'])
parser.add_argument('--all_subject', action='store_true')
parser.add_argument('--get_prediction', default=False) # action='store_true')

parser.add_argument('--data-root', default='./bcic4-2a')
parser.add_argument('--save-root', default='./pp/') # lr=0.005, wd=0.001, seed=2021
parser.add_argument('--result-dir', default='/')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 8)')
parser.add_argument('--epochs', type=int, default=3000, metavar='N', # 1000
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.005, metavar='LR',
                    help='learning rate (default: 0.05)')
parser.add_argument('--save-model', action='store_true', default=True, help='For Saving the current best Model')
parser.add_argument('--seed', default=2021, help='Seed value')
# parser.add_argument('--gamma', default=0.995, help='Seed value')
parser.add_argument('--weight_decay', default=0.01, help='adam optimizer weight decay')
parser.add_argument('--generate', default=False, help='use generated data')

# PARAMETER 불러오기
args = parser.parse_args()

# main_resnet.main(args)
alpha=0
main_deep.main(args, alpha)
main_shallow.main(args, alpha)
# p.main(args,alpha)

# alpha=0.3
# main_deep.main(args, alpha)
# main_shallow.main(args, alpha)