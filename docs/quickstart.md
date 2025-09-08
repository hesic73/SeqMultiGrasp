# Quickstart

## grasp_generation

### Single-Object Grasp Pose Synthesis

```bash
cd grasp_generation

python main.py \
    object_code=cube \
    name=my_run_cube \
    'active_links=["link_13.0","link_14.0","link_15.0","link_15.0_tip","link_1.0","link_2.0","link_3.0","link_3.0_tip"]' \
    n_contact=2 \
    n_iter=6000 \
    batch_size=256
```

The results will be saved under `data/experiments/my_run_cube`. TensorBoard logs are located in the `logs` subdirectory:

```bash
tensorboard --logdir data/experiments/my_run_cube/logs
```

In the `results` folder, you will find several `.npy` files containing hand configurations:

- `{object_code}.npy`: Raw DFC optimization results, including the starting state.
- `{object_code}_success.npy`: DFC optimization results that meet certain criteria.
- `{object_code}_validated.npy`: Hand poses that pass floating-hand validation in simulation, as in [DexGraspNet](https://pku-epic.github.io/DexGraspNet/).
- `{object_code}_tabletop_validated.npy`: Hand poses that pass tabletop validation in simulation, as in [Get a Grip](https://sites.google.com/view/get-a-grip-dataset).

You can use the visualization scripts to view the results.


Similarly, we can generate poses for side grasps:

```bash
python main.py \
    object_code=cylinder_r_2_85_h_10_5 \
    name=my_run_cylinder_r_2_85_h_10_5 \
    'active_links=["link_9.0","link_10.0","link_11.0","link_11.0_tip","base_link"]' \
    n_contact=4 \
    n_iter=6000 \
    batch_size=256 \
    initialization_method=side_grasp \
    'qpos_loc=[0.0,-0.196,0.0,0.4,0.0,0.7,0.0,0.4,0.021,1.395,1.05,0.797,0.263,0.0,0.0,0.0]' \
    ground_offset=0.02
```

### Merge Grasps


```bash
python merge.py --path_0 ../data/experiments/my_run_cube/results/cube_success_validated.npy \
--path_1 ../data/experiments/my_run_cylinder_r_2_85_h_10_5/results/cylinder_r_2_85_h_10_5_tabletop_validated.npy \
--save_path cube_cylinder_r_2_85_h_10_5.h5
```


### Visualization

Use the new Python scripts with tyro CLI:

```bash
python visualize_single_object.py \
  --data_path ../data/experiments/my_run_cylinder_r_2_85_h_10_5/results/cylinder_r_2_85_h_10_5_tabletop_validated.npy \
  --index 0

python visualize_multiple_objects.py \
  --hdf5_path cube_cylinder_r_2_85_h_10_5.h5 \
  --index 0
```

### Evaluation

Run grasp attempts sampled from a merged HDF5 file:

```bash
python eval.py \
  --data_path cube_cylinder_r_2_85_h_10_5.h5 \
  --n 1 \
  --vis \
  --output_dir "../data/experiments/eval_cube_cylinder_r_2_85_h_10_5"
```