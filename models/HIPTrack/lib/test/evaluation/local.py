from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/home/vli/thangdd_workspace/MOT/data/got10k_lmdb'
    settings.got10k_path = '/home/vli/thangdd_workspace/MOT/data/'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/home/vli/thangdd_workspace/MOT/data/itb'
    settings.lasot_extension_subset_path_path = '/home/vli/thangdd_workspace/MOT/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/home/vli/thangdd_workspace/MOT/data/lasot_lmdb'
    settings.lasot_path = '/home/vli/thangdd_workspace/MOT/data/lasot'
    settings.network_path = '/home/vli/thangdd_workspace/MOT/models/HIPTrack/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/home/vli/thangdd_workspace/MOT/data/nfs'
    settings.otb_path = '/home/vli/thangdd_workspace/MOT/data/otb'
    settings.prj_dir = '/home/vli/thangdd_workspace/MOT/models/HIPTrack'
    settings.result_plot_path = '/home/vli/thangdd_workspace/MOT/models/HIPTrack/output/test/result_plots'
    settings.results_path = '/home/vli/thangdd_workspace/MOT/models/HIPTrack/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/home/vli/thangdd_workspace/MOT/models/HIPTrack/'
    settings.segmentation_path = '/home/vli/thangdd_workspace/MOT/models/HIPTrack/output/test/segmentation_results'
    settings.tc128_path = '/home/vli/thangdd_workspace/MOT/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/home/vli/thangdd_workspace/MOT/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/home/vli/thangdd_workspace/MOT/data/trackingnet'
    settings.uav_path = '/home/vli/thangdd_workspace/MOT/data/uav'
    settings.vot18_path = '/home/vli/thangdd_workspace/MOT/data/vot2018'
    settings.vot22_path = '/home/vli/thangdd_workspace/MOT/data/vot2022'
    settings.vot_path = '/home/vli/thangdd_workspace/MOT/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

