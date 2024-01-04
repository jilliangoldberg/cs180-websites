import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as sktr
import skimage.io as skio
import skimage as sk
from matplotlib.path import Path

def load_image(path, as_gray=False, folder='images/'):
    im = skio.imread(folder + path, as_gray=as_gray)
    im = sk.img_as_float(im)
    skio.imshow(im)
    return im

def scale_mask_to_dest(src_pts, dest_pts):
    # create a transformation matrix to map destination points to a rectangle
    src_rect = np.array([src_pts[0], src_pts[3], src_pts[5], src_pts[8]])

    # apply the transformation to the source points
    transform = sktr.ProjectiveTransform()
    transform.estimate(src_rect, dest_pts)
    transformed_pts = transform(src_pts)

    return transformed_pts

def get_mask(src_im, dest_im):
    im_height, im_width, _ = src_im.shape
    dest_height, dest_width, _ = dest_im.shape

    print('draw shape around object, q to stop')
    plt.figure()
    plt.imshow(src_im)

    # select points
    src_pts = plt.ginput(10, timeout=500)
    src_pts = np.array(src_pts)
    plt.close()
    plt.figure()
    plt.imshow(src_im)
    plt.plot(src_pts[:,0], src_pts[:,1], "*-")
    plt.show()

    # choose roughly where to place mask
    plt.figure()
    plt.imshow(dest_im)
    plt.show()
    print('click 4 points to place mask')
    dest_pts = plt.ginput(4, timeout=500)
    dest_pts = np.array(dest_pts)
    plt.close()
    plt.figure()
    plt.imshow(dest_im)
    plt.plot(dest_pts[:,0], dest_pts[:,1], "*-")
    plt.show()

    # scale points to destination image
    transformed_pts = scale_mask_to_dest(src_pts, dest_pts)
    
    # make mask from points
    path = Path(transformed_pts)
    y, x = np.mgrid[:dest_height, :dest_width]
    points = np.column_stack((x.ravel(), y.ravel()))
    mask = path.contains_points(points).reshape((dest_height, dest_width))
    
    return mask, transformed_pts, dest_pts
    # return mask, poly

def improved_get_mask(src_im, dest_im):
    im_height, im_width, _ = src_im.shape
    dest_height, dest_width, _ = dest_im.shape

    # choose roughly where to place mask
    plt.figure()
    plt.imshow(dest_im)
    plt.show()
    print('click the top and bottom middle points to place object')
    dest_pts = plt.ginput(2, timeout=500)
    dest_pts = np.array(dest_pts)
    plt.close()

    # scale object to selected points
    topx, topy, bottomx, bottomy = dest_pts[0,0], dest_pts[0,1], dest_pts[1,0], dest_pts[1,1]
    goal_height = bottomy - topy
    ratio = goal_height/im_height
    im_resized = sktr.resize(src_im, (int(goal_height), int(im_width*ratio)))
    new_height, new_width, _ = im_resized.shape

    # give object im padding to match mask placement
    src_im_padded = np.zeros((dest_height, dest_width, 3))
    halfgoalh, halfgoalw = int(new_height/2), int(new_width/2)
    try:
        src_im_padded[int(bottomy-new_height):int(bottomy), int(topx-halfgoalw):int(topx+halfgoalw)] = im_resized
    except:
        src_im_padded[int(bottomy-new_height):int(bottomy), int(topx-halfgoalw):int(topx+halfgoalw)+1] = im_resized
    plt.imshow(src_im_padded, cmap='gray')

    # now select mask for object!
    print('draw shape around object, q to stop')
    plt.figure()
    plt.imshow(src_im_padded)

    # select points
    src_pts = plt.ginput(10, timeout=500)
    src_pts = np.array(src_pts)
    plt.close()
    plt.figure()
    plt.imshow(src_im_padded)
    plt.plot(src_pts[:,0], src_pts[:,1], "*-")
    plt.show()
    
    # make mask from points
    path = Path(src_pts)
    y, x = np.mgrid[:dest_height, :dest_width]
    points = np.column_stack((x.ravel(), y.ravel()))
    mask = path.contains_points(points).reshape((dest_height, dest_width))
    
    return mask, src_pts, dest_pts, src_im_padded