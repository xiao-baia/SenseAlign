from modelscope import snapshot_download
SenseVoiceSmall_Model = snapshot_download('iic/SenseVoiceSmall', cache_dir='./models/iic/SenseVoiceSmall')
Speech_FSMN_Vad_Model = snapshot_download('iic/speech_fsmn_vad_zh-cn-16k-common-pytorch', cache_dir='./models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch')
print('Done!')