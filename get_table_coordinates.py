from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
# Load model

def return_table_coordinates(out_file_name):
	config_file = '/content/CascadeTabNet/Config/cascade_mask_rcnn_hrnetv2p_w32_20e.py'
	checkpoint_file = '/content/epoch_36.pth'

	model = init_detector(config_file, checkpoint_file, device='cuda:0')

	# Run Inference
	result = inference_detector(model, out_file_name)
	# Visualization results
	show_result_pyplot(out_file_name, result,('Bordered', 'cell', 'Borderless'), score_thr=0.85)
	return result