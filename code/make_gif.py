import imageio

def gen_gif(imgs_path, save_path):

    images = []
    for img_path in imgs_path:
        images.append(imageio.imread(img_path))
    imageio.mimsave(save_path, images)

path = '/Users/phil/GitHub/Academic/Courses/IFT6266_H17/IFT6266_H17-project/results/20170220_sbatch_33410/im01/'
file_paths = []
nb_epochs = 23
for x in xrange(1, nb_epochs + 1):
    file_path = path + 'img_epoch_%s_id_1_pred.jpg' % x
    file_paths.append(file_path)
save = '/Users/phil/GitHub/Academic/Courses/IFT6266_H17/IFT6266_H17-project/results/20170220_sbatch_33410/gifs/im01.gif'
gen_gif(file_paths, save)


path = '/Users/phil/GitHub/Academic/Courses/IFT6266_H17/IFT6266_H17-project/results/20170220_sbatch_33410/im02/'
file_paths = []
nb_epochs = 23
for x in xrange(1, nb_epochs + 1):
    file_path = path + 'img_epoch_%s_id_2_pred.jpg' % x
    file_paths.append(file_path)
save = '/Users/phil/GitHub/Academic/Courses/IFT6266_H17/IFT6266_H17-project/results/20170220_sbatch_33410/gifs/im02.gif'
gen_gif(file_paths, save)

path = '/Users/phil/GitHub/Academic/Courses/IFT6266_H17/IFT6266_H17-project/results/20170220_sbatch_33410/im03/'
file_paths = []
nb_epochs = 23
for x in xrange(1, nb_epochs + 1):
    file_path = path + 'img_epoch_%s_id_3_pred.jpg' % x
    file_paths.append(file_path)
save = '/Users/phil/GitHub/Academic/Courses/IFT6266_H17/IFT6266_H17-project/results/20170220_sbatch_33410/gifs/im03.gif'
gen_gif(file_paths, save)
