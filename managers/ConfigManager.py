import json
import os
import sys


class ConfigManager:
    _instance = None
    
    @staticmethod
    def get_instance():
        if ConfigManager._instance is None:
            ConfigManager._instance = ConfigManager()
        return ConfigManager._instance
    def __init__(self):
        if ConfigManager._instance is not None:
            raise Exception("ConfigManager is a singleton!")
            return
        else:
            ConfigManager._instance = self

            self.defaults = {
                    "LOG_LEVEL": 10,
                    "LOG_TO_CONSOLE": False,
                    "CONF_THRESHOLD": 0.25,
                    "IOU_THRESHOLD": 0.7,
                    "IMGSZ":[640,640],
                    "ALIVE_COUNT": 5,
                    "MISSING_COUNTER": 10,
                    "CAMERAS_PER_PROCESS": 3,
                    "IS_VIDEO": False,
                    "IS_LIVE": False,
                    "VIDEO_PATH": "/home/wot-keval/Music/recording_20250814_125141.avi",
                    "URL": "https://ams.weboccult.com",
                    "BUILD_VERSION": "v1",
                    "MODEL_FOLDERS":"MODELS",
                    "DETECTION_MODEL_NAME":"best4.onnx",
                    "S3_FOLDER": "INTRUSION",
                    "CAMERA_DETAILS_JSON_FILE_NAME": "camera_details.json",
                    "Model_folder": "Models",
                    "DETECTION_MODEL_PATH":"WOT03-det-640-20251013.bin",  
                    "INTRUSION_ALERT_INTERVAL_NORMAL": 5, # in seconds
                    "INTRUSION_ALERT_INTERVAL_CRITICAL": 3, # in seconds
                    "ROI_UPDATE_CHECK_INTERVAL_TIME": 300, # in seconds
                    'DEBUG_MODE':True,
                    "ALERT_INTERVAL_SECONDS":300,
                    "ENCRYPTED_MODELS_PATH":"_internal/cache/models",
                    "ENCRYPTED_FILE_PATH":"_internal/cache/temp.txt",
                    "TOKEN_API_URL":"https://app.gotilo.ai/api/v1/tokens/verify",
                    "GET_DETVICE_DATA_API_URL":"https://app.gotilo.ai",
                    "DETECTION_MODEL_TYPE":"onnx",
                    "ONNX_PROVIDER":[
    ('TensorrtExecutionProvider', {
        'device_id': 0,                       
        'trt_fp16_enable': True,      
        'trt_engine_cache_enable':True,
        'trt_engine_cache_path':'Models'       
    }),
    ('CUDAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
        'cudnn_conv_algo_search': 'EXHAUSTIVE',
        'do_copy_in_default_stream': True,
    })
],
                    "RABBITMQ_WRITER_QUEUE_SIZE":500,
                    "HEARTBEAT_DEVICE":True,
                    "CAMERA_HEARTBEAT_TIMEOUT_MINS":2,
                    "RABBITMQ_QUEUE": "intrusion_event_logs",
                    "INTRUSION_DETECTED_BUFFER_NORMAL":5,
                    "INTRUSION_DETECTED_BUFFER_CRITICAL":2,
            }
            self.private = {
                "RABBITMQ_HOST": "localhost",
                "RABBITMQ_PORT": "5672",
                "RABBITMQ_USERNAME": "keval",
                "RABBITMQ_PASSWORD": "2232",
                "RABBITMQ_EXCHANGE": "intrusion_event_logs",
                "RABBITMQ_ROUTING_KEY": "intrusion_event_logs",
                "RABBITMQ_WRITER_QUEUE_SIZE": 500,
                
                
    
        }
            
        self.read_config_file()
            

    def read_config_file(self):
        if os.path.exists("config.json"):
            config_file = open("config.json","r")
            config = json.load(config_file)
            self.defaults.update(config)
            config_file.close()
        else:
            self.save_config_file()

    def save_config_file(self):
        pass
        # config_file = open("config.json","w")

        # json.dump(self.defaults,config_file,indent=4)
        # config_file.close()

    def set(self,key,value):
        
        self.defaults[key] = value
        self.save_config_file()
        
    def set_all(self,settings):
        self.defaults.update(settings)
        self.save_config_file()

    def get(self,key,default=None):
        if key in self.defaults:
            return self.defaults[key]
        elif key in self.private:
            return self.private[key]
        else:
            return default
        
    def get_private(self,key,default=None):

        if key in self.private:
            return self.private[key]
        else:
            return default

    def set_private(self,key,value):
        self.private[key] = value

if __name__=="__main__":
    configmanager = ConfigManager.get_instance()
    print(configmanager.get("IS_LIVE"))

       
