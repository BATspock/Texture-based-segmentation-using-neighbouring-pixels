import cv2
import sys
import numpy as np

class preprocessImageWithKernles(object):
    """
    create vector features after applying kernels
    """
    def __init__(self, image, image_color):
        #convert image to grayscale
        self.im = image  
        self.im_color = image_color
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  

    def merge(self):
        """
        apply gaussian blur
        """
        self.out = cv2.GaussianBlur(self.im,(7,7),0)
        self.out_color = cv2.GaussianBlur(self.im_color,(7,7),0)

    def apply_kernel(self, kernels):
        """
        apply LM filter kernels on the image after preprocessing
        """
        image = []
        
        for i in range(kernels.shape[2]):
            image.append(cv2.filter2D(self.out, -1, kernels[:,:,i]))

        self.image_after_filters = np.array(image)
        #print("creating filtered image...")
        #print(self.image_after_filters.shape)
        
    def create_vectors(self):
        """
        return vectors created for each pixel
        """
        point_vector = list()
        for row in range(self.image_after_filters.shape[1]):
            for col in range(self.image_after_filters.shape[2]):
                point_vector.append(self.image_after_filters[:,row, col])
        pixel_point_vector = np.array(point_vector)
        #print(self.pixel_point_vector.shape)
        #print("initial vector...")
        return pixel_point_vector

    def add_neighbour_info(self, kernel_size):
        """
        add information of neighbouring pixels using a k X k mask
        """
        pixel_point_vector = self.create_vectors()
        extra_info = list()
        for row in range(0, self.image_after_filters.shape[1]):
            for col in range(0, self.image_after_filters.shape[2]): 
                interm_hold_values = list()               
                for x in range(row-kernel_size,row+kernel_size):
                    for y in range(col - kernel_size, col + kernel_size):
                        try:
                            if(x==row and y==col):
                                pass
                            else:
                                interm_hold_values.append(self.image_after_filters[:,x,y])
                        except IndexError:
                            interm_hold_values.append([0]*48)
                extra_info.append(interm_hold_values)

        extra_info = np.array(extra_info)
        #print(extra_info.shape)
        extra_info = np.reshape(extra_info,(extra_info.shape[0], extra_info.shape[1]*extra_info.shape[2]))
        final_out = np.concatenate((pixel_point_vector, extra_info), axis=1)
        return final_out, self.out_color

        
