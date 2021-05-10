import hpacellseg.cellsegmentator as cellsegmentator
from hpacellseg.utils import label_cell, label_nuclei
import cv2
import numpy as np
import pandas as pd

def rle_to_mask(rle_string, height, width):
    """ Convert RLE sttring into a binary mask 
    
    Args:
        rle_string (rle_string): Run length encoding containing 
            segmentation mask information
        height (int): Height of the original image the map comes from
        width (int): Width of the original image the map comes from
    
    Returns:
        Numpy array of the binary segmentation mask for a given cell
    """
    rows,cols = height,width
    rle_numbers = [int(num_string) for num_string in rle_string.split(' ')]
    rle_pairs = np.array(rle_numbers).reshape(-1,2)
    img = np.zeros(rows*cols,dtype=np.uint8)
    for index,length in rle_pairs:
        index -= 1
        img[index:index+length] = 255
    img = img.reshape(cols,rows)
    img = img.T
    return img

def get_contour_bbox_from_rle(rle, width, height, return_mask=True,):
    """ Get bbox of contour as `xmin ymin xmax ymax`
    
    Args:
        rle (rle_string): Run length encoding containing 
            segmentation mask information
        height (int): Height of the original image the map comes from
        width (int): Width of the original image the map comes from
    
    Returns:
        Numpy array for a cell bounding box coordinates
    """
    mask = rle_to_mask(rle, height, width).copy()
    cnts = grab_contours(
        cv2.findContours(
            mask, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        ))
    x,y,w,h = cv2.boundingRect(cnts[0])
    
    if return_mask:
        return (x,y,x+w,y+h), mask
    else:
        return (x,y,x+w,y+h)

def grab_contours(cnts):
    # if the length the contours tuple returned by cv2.findContours
    # is '2' then we are using either OpenCV v2.4, v4-beta, or
    # v4-official
    if len(cnts) == 2:
        cnts = cnts[0]

    # if the length of the contours tuple is '3' then we are using
    # either OpenCV v3, v4-pre, or v4-alpha
    elif len(cnts) == 3:
        cnts = cnts[1]

    # otherwise OpenCV has changed their cv2.findContours return
    # signature yet again and I have no idea WTH is going on
    else:
        raise Exception(("Contours tuple must have length 2 or 3, "
            "otherwise OpenCV changed their cv2.findContours return "
            "signature yet again. Refer to OpenCV's documentation "
            "in that case"))

    # return the actual contours array
    return cnts

def get_contour_bbox_from_raw(raw_mask):
    """ Get bbox of contour as `xmin ymin xmax ymax`
    
    Args:
        raw_mask (nparray): Numpy array containing segmentation mask information
    
    Returns:
        Numpy array for a cell bounding box coordinates
    """
    cnts = grab_contours(
        cv2.findContours(
            raw_mask, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        ))
    xywhs = [cv2.boundingRect(cnt) for cnt in cnts]
    xys = [(xywh[0], xywh[1], xywh[0]+xywh[2], xywh[1]+xywh[3]) for xywh in xywhs]
    return sorted(xys, key=lambda x: (x[1], x[0]))


def pad_to_square(a):
    """ Pad an array `a` evenly until it is a square """
    if a.shape[1]>a.shape[0]: # pad height
        n_to_add = a.shape[1]-a.shape[0]
        top_pad = n_to_add//2
        bottom_pad = n_to_add-top_pad
        a = np.pad(a, [(top_pad, bottom_pad), (0, 0), (0, 0)], mode='constant')

    elif a.shape[0]>a.shape[1]: # pad width
        n_to_add = a.shape[0]-a.shape[1]
        left_pad = n_to_add//2
        right_pad = n_to_add-left_pad
        a = np.pad(a, [(0, 0), (left_pad, right_pad), (0, 0)], mode='constant')
    else:
        pass
    return a


def cut_out_cells(rgby, rles, resize_to=(256,256), square_off=True, return_masks=False, from_raw=True):
    """ Cut out the cells as padded square images 
    
    Args:
        rgby (np.array): 4 Channel image to be cut into tiles
        rles (list of RLE strings): List of run length encoding containing 
            segmentation mask information
        resize_to (tuple of ints, optional): The square dimension to resize the image to
        square_off (bool, optional): Whether to pad the image to a square or not
        
    Returns:
        list of square arrays representing squared off cell images
    """
    w,h = rgby.shape[:2]
    contour_bboxes = [get_contour_bbox_from_rle(rle, w, h, return_mask=return_masks) for rle in rles]
    if return_masks:
        masks = [x[-1] for x in contour_bboxes]
        contour_bboxes = [x[:-1] for x in contour_bboxes]
    
    arrs = [rgby[bbox[1]:bbox[3], bbox[0]:bbox[2], ...] for bbox in contour_bboxes]
    if square_off:
        arrs = [pad_to_square(arr) for arr in arrs]
        
    if resize_to is not None:
        arrs = [
            cv2.resize(pad_to_square(arr).astype(np.float32), 
                       resize_to, 
                       interpolation=cv2.INTER_CUBIC) \
            for arr in arrs
        ]
    if return_masks:
        return arrs, masks
    else:
        return arrs


# Assuming that there are images in the current folder with the
# following names.
images = [
    ["microtubules_one.tif", "microtubules_two.tif"],
    ["endoplasmic_reticulum_one.tif", "endoplasmic_reticulum_two.tif"],
    ["nuclei_one.tif", "nuclei_two.tif"]
]
NUC_MODEL = "./nuclei-model.pth"
CELL_MODEL = "./cell-model.pth"
segmentator = cellsegmentator.CellSegmentator(
    NUC_MODEL,
    CELL_MODEL,
    scale_factor=0.25,
    device="cuda",
    # NOTE: setting padding=True seems to solve most issues that have been encountered
    #       during our single cell Kaggle challenge.
    padding=False,
    multi_channel_model=True,
)

# For nuclei
nuc_segmentations = segmentator.pred_nuclei(images[2])

# For full cells
cell_segmentations = segmentator.pred_cells(images)

# post-processing
nuclei_mask = label_nuclei(nuc_segmentations[0])
nuclei_mask, cell_mask = label_cell(nuc_segmentations[1], cell_segmentations[1])