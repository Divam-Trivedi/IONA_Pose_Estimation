from Utils import *

# -------------------- User Config --------------------
Folder_Name = 'Meal_Tray_Scenario'
model_name = 'medical_objects_fruits'
target_classes = ["apple", "banana", "yogurt", "sb_cup", "ketchup", "tray", "orange"]
script_directory = pathlib.Path(__file__).parent.resolve()
ROOT = pathlib.Path(f"{script_directory}/{Folder_Name}")

MAX_FRAMES_DEFAULT = 5
# -----------------------------------------------------

def main():

    camera_intrinsics = setup_intelrealsense_camera()
    all_classes, dataset_name, mesh_names = parse_model_info(model_name)
    predictor = build_detector(dataset_name, len(all_classes), all_classes,
                               model_path=f"{script_directory}/detectron2_models/{model_name}.pth")
    
    frames_rgb, frames_depth, masks_per_frame, detected_objects, start_time, persistent_objects = capture_images(camera_intrinsics, predictor, dataset_name, target_classes, MAX_FRAMES=MAX_FRAMES_DEFAULT)
    
    detected_objects = set()
    instance_to_class = {}
    for frame_mask_dict in masks_per_frame:
        for iid in frame_mask_dict.keys():
            detected_objects.add(iid)
            instance_to_class[iid] = persistent_objects[iid]['class']
    end_time = estimate_pose(frames_rgb, frames_depth, masks_per_frame, detected_objects, camera_intrinsics, mesh_names, all_classes, ROOT, instance_to_class)

    print(f"Total processing time: {end_time - start_time:.2f} seconds for {len(detected_objects)} object(s)")

if __name__ == "__main__":
    main()
