import argparse
import os
import lpips

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dir0', type=str, default='/media/jty/5cbf9bee-8562-4da0-9e5a-10ffac7458a9/JTY/testdataset/outdoor/hazy')
parser.add_argument('--dir1', type=str, default='/media/jty/5cbf9bee-8562-4da0-9e5a-10ffac7458a9/JTY/testdataset/outdoor/gt')
parser.add_argument('-v', '--version', type=str, default='0.1')
opt = parser.parse_args()

## Initializing the model
loss_fn = lpips.LPIPS(net='alex', version=opt.version)

# the total list of images
files = os.listdir(opt.dir0)
i = 0
total_lpips_distance = 0
average_lpips_distance = 0
for file in files:

    try:
        # Load images
        img0 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir0, file)))
        img1 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir1, file)))

        if (os.path.exists(os.path.join(opt.dir0, file)), os.path.exists(os.path.join(opt.dir1, file))):
            i = i + 1

        # Compute distance
        current_lpips_distance = loss_fn.forward(img0, img1)
        total_lpips_distance = total_lpips_distance + current_lpips_distance

        print('%s: %.3f' % (file, current_lpips_distance))

    except Exception as e:
        print(e)

average_lpips_distance = float(total_lpips_distance) / 500

print("The processed iamges is ", i, "and the average_lpips_distance is: %.3f" % average_lpips_distance)