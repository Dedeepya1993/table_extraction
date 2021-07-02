import numpy as np
import cv2
import fitz
def save_image_from_pdf_page(pdf,page_number,out_file_name):
	doc = fitz.open(pdf)
	page = doc.loadPage(page_number)
	trans = fitz.Matrix(1, 1).preRotate(0)
	pm = page.getPixmap(matrix=trans, alpha=False)
	                # Get the data stream here, look at the source code, you can use getPNGdata() directly below
	getpngdata = pm.getImageData(output="png")
	                # Decode to np.uint8
	image_array = np.frombuffer(getpngdata, dtype=np.uint8)
	img_cv = cv2.imdecode(image_array, cv2.IMREAD_ANYCOLOR)
	cv2.imwrite(out_file_name,img_cv)