import numpy as np
import cv2


class PatchFactory(object):
    def __init__(self, data_engine):
        self.data_engine = data_engine

    def _get_hist(self, image):
        image_hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(image_hist, image_hist)
        return image_hist.flatten()
        
    def _find_similar(self, query, dataset):
        score = 0
        #query = cv2.cvtColor(query, cv2.COLOR_BGR2RBG)
        query_hist = self._get_hist(query)
        patch_ind = None
        for i in range(len(dataset)):
            patch_hist = self._get_hist(dataset[i])
            score_buf = cv2.compareHist(query_hist, patch_hist, cv2.HISTCMP_INTERSECT) 
            if score < score_buf:
                score = score_buf
                patch_ind = i
        return None if patch_ind is None else dataset[patch_ind]

    def recreate_image(self, image):
        self.data_engine.download_dataset()
        dataset = self.data_engine.get_images()
        row_count = int(image.shape[0]/self.data_engine.image_size[0])
        col_count = int(image.shape[1]/self.data_engine.image_size[1])
        hi_rez_resize_to = (self.data_engine.image_size[0] * row_count,
                            self.data_engine.image_size[1] * col_count)
        new_image = cv2.resize(image, hi_rez_resize_to)

        for i in range(row_count):
            for j in range(col_count):
                print (i, j), (row_count, col_count)
                tile = image[i * self.data_engine.image_size[0]: (i + 1) * self.data_engine.image_size[0],
                             j * self.data_engine.image_size[1]: (j + 1) * self.data_engine.image_size[1], :]
                replace_with = self._find_similar(tile, dataset)
                image[i * self.data_engine.image_size[0]: (i + 1) * self.data_engine.image_size[0],
                      j * self.data_engine.image_size[1]: (j + 1) * self.data_engine.image_size[1], :] = replace_with
        cv2.imwrite('iPatched.png', image)
