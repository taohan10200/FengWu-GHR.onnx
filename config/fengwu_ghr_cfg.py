onnx_dir = 'onnx/fengwu_ghr/onnx_dir/'
onnx_keys = ['encoder'] + [f'block_{i}' for i in range(0,30)]  + ['decoder'] #['encoder'] + 
poolsize_GB = 72     #'onnxruntime memory pool size. default value is 32GB'
temperature = 0.1    #'factor to scale up logits, 1.0 means no warp. use `0.1` by default.')
topk = 40             # 'filter k high score values from logits, None means no filter. 40 by default.'
fp16 = True           #'enable fp16 inference, default True.'
inference_steps = 40 # one step is 6 hour interval

total_levels= [1000.,  975.,  950.,  925.,  900.,  875.,  850.,  825.,  800.,
 775.,  750.,  700.,  650.,  600.,  550.,  500.,  450.,  400.,
 350.,  300.,  250.,  225.,  200.,  175.,  150.,  125.,  100.,
 70.,   50.,   30.,   20.,   10.,    7.,    5.,    3.,    2.,
 1.]

vnames=dict(
    pressure=['z','q', 'u', 'v', 't'],
    single=['v10','u10','v100', 'u100', 't2m','tcc', 'sp','tp6h', 'msl']) #'tisr',


pressure_level = [1000.,  925.,  850.,  700.,    600.,   500.,   400.,
                  300.,  250.,    200.,   150.,    100.,
                  50.]


save_cfg = dict(

                s3_cfg =dict(
                    internal_ak_sk = 'ai4earth',
                    bucket_name='nwp_predictions',
                    endpoint='http://10.140.31.254'),
                save_path='./data/output' ,   
                variables_list =['z_1000','z_850','z_500','z_100','z_50',
                                        'q_1000','q_850','q_500','q_100','q_50',
                                        'u_1000','u_850','u_500','u_100','u_50',
                                        'v_1000','v_850','v_500','v_100','v_50',
                                        't_1000','t_850','t_500','t_100','t_50',
                                        'v10','u10','v100', 'u100', 't2m','tcc', 'sp','tp6h', 'msl']
                )