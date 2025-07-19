import skimage
import os
import numpy as np
import scipy
import glob


# def get_patches(path):
# 	return os.listdir(path)


def get_patch_info(path, name):
	patches = glob.glob(path+name+'*.*')
	img = skimage.color.gray2rgb(skimage.io.imread(patches[0]))
	max_x,max_y=1,1
	for name in patches:
		if(int(name.split('_')[-2]) > max_x) : max_x = int(name.split('_')[-2])
		if(int(name.split('_')[-1].split('.')[0]) > max_y) : max_y = int(name.split('_')[-1].split('.')[0])
	# shapesize = scipy.ndimage.rotate(skimage.io.imread(path+patches[0]), 270, order=0).shape
	shapesize = skimage.color.gray2rgb(skimage.io.imread(patches[0])).shape
	c = -1
	try:
		c = shapesize[2]
	except:
		pass
	return {'max_x': max_x +1, 'max_y': max_y +1, 'patchsize': (shapesize[0],shapesize[1],c), 
	'suffix': patches[0].split('.')[1], 'patcheslist': patches}


def get_img(path_patchname):
	# return scipy.ndimage.zoom(skimage.color.gray2rgb(skimage.io.imread(path_patchname)), [0.1, 0.1, 1], order=0)
	return scipy.ndimage.zoom(skimage.io.imread(path_patchname), [0.1, 0.1, 1], order=0)
	# return scipy.ndimage.rotate(skimage.io.imread(path+patchname), 270, order=0)


def createimg(x1, x2, y1, y2, path, name):
	info=get_patch_info(path, name)
	max_x, max_y, h, w = info['max_x'], info['max_y'], info['patchsize'][0]//10, info['patchsize'][1]//10
	if(x1>x2): x1, x2 = x2, x1
	if(y1>y2): y1, y2 = y2, y1
	if(x1>max_x): x1 = max_x
	if(x2>max_x): x2 = max_x
	if(y1>max_y): y1 = max_y
	if(y2>max_y): y2 = max_y
	if(info['patchsize'][2] == -1):
		res = np.ones((int(info['patchsize'][0])*(x2-x1+1)//10, int(info['patchsize'][1])*(y2-y1+1))//10) * 0.6
	else:
		res = np.zeros((int(info['patchsize'][0])*(x2-x1+1)//10, int(info['patchsize'][1])*(y2-y1+1)//10, int(info['patchsize'][2])))
		res[:,:,1:2] = 99
	for i in range(x1, x2+1):
		for j in range(y1, y2+1):
			path_patchname = '_'.join(str(i) for i in info['patcheslist'][0].split('_')[:-2])+'_'+str(i)+'_'+str(j)+'.'+info['suffix']
			if(path_patchname in info['patcheslist']): 
				res[(i-x1)*h:(i-x1+1)*h, (j-y1)*w:(j-y1+1)*w] = get_img(path_patchname).copy()
	return res


# if __name__ == '__main__':
# 	# path = '/data4T_1/huangruiwei/data/testoutput/18-29252/'
# 	# outpath = '/data4T_1/huangruiwei/data/testoutput/'
# 	path = '/home/sda/huangruiwei/data/slides666output/'
# 	outpath = '/home/sda/huangruiwei/data/slides666output_patchimg/'
# 	filelist = os.listdir(path)
# 	zoom = 0.1
# 	x1, y1 = 100, 100
# 	x2, y2 = 0, 0
# 	os.makedirs(outpath, exist_ok=True)
# 	for name in filelist:
# 		res = createimg(x1, x2, y1, y2, path+name+'/')
# 		smallimg = scipy.ndimage.zoom(res, [zoom, zoom, 1], order=0)
# 		skimage.io.imsave(outpath+name+'('+str(x1)+','+str(y1)+'-'+str(x2)+','+str(y2)+').jpg',smallimg)
# 		print(name+' done!')


path = '/home/sdb/huangruiwei/data/camelyon16_patch/imgs1024_train/'
outpath = '/home/sdb/huangruiwei/data/camelyon16_patch/'
zoom = 0.05
x1, y1 = 301, 301
x2, y2 = 0, 0

filenames = os.listdir(path)
filelist = []
for name in filenames:
	# filelist.append(name.split('_')[0])
	filelist.append(name.split('_')[0] + '_' + name.split('_')[1])


filelist = sorted(list(set(filelist)))
os.makedirs(outpath, exist_ok=True)
for name in filelist:
	res = createimg(x1, x2, y1, y2, path, name)
	# smallimg = scipy.ndimage.zoom(res, [zoom * 10, zoom * 10, 1], order=0)
	smallimg = scipy.ndimage.zoom(res, [zoom * 10, zoom * 10, 1], order=0).astype(np.uint8)
	skimage.io.imsave(outpath+name+'('+str(x1)+','+str(y1)+'-'+str(x2)+','+str(y2)+').jpg',smallimg)
	#res = createimg(x1, x2, y1, y2, path+'masks/', name)
	#smallimg = scipy.ndimage.zoom(res, [zoom * 10, zoom * 10, 1], order=0)
	#skimage.io.imsave(outpath+name+'('+str(x1)+','+str(y1)+'-'+str(x2)+','+str(y2)+').png',smallimg)
	print(name+' done!')











