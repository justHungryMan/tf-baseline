import requests
import time
import logging
import tensorflow as tf

LOGGER = logging.getLogger(__name__)

class TPUManager:
    def __init__(self, project='TPU0'):
        self.base_url = 'https://tpu.googleapis.com'
        self.project = 'TPU0'
        self.zone = 'europe-west4-a'

    @property
    def headers(self):
        return {
            'Authorization': f'Bearer {self.get_access_token_internal()}',
        }

    @staticmethod
    def get_access_token_internal():
        METADATA_URL = 'http://metadata.google.internal/computeMetadata/v1/'
        METADATA_HEADERS = {'Metadata-Flavor': 'Google'}
        SERVICE_ACCOUNT = 'default'

        url = '{}instance/service-accounts/{}/token'.format(
            METADATA_URL, SERVICE_ACCOUNT)

        # Request an access token from the metadata server.
        r = requests.get(url, headers=METADATA_HEADERS)
        r.raise_for_status()

        # Extract the access token from the response.
        return r.json()['access_token']

    def create(self, tpu_name, tpu_type='v3-8', sync=True):
        data = {
            "acceleratorType": tpu_type,
            "tensorflowVersion": self.version(),
            'schedulingConfig': {
                'preemptible': False, #tpu_type.endswith('-8')
            },
        }

        url = f'{self.base_url}/v1alpha1/projects/{self.project}/locations/{self.zone}/nodes?nodeId={tpu_name}'

        r = requests.post(url, json=data, headers=self.headers)
        self._raise_for_status(r)

        # print(f'Creating {tpu_name}...')
        if sync:
            self._wait(tpu_name, 'READY')

    def version(self):
        return tf.__version__

    def list(self):
        url = f'{self.base_url}/v1alpha1/projects/{self.project}/locations/{self.zone}/nodes'

        r = requests.get(url, headers=self.headers)
        self._raise_for_status(r)

        rlist = [[x['name'], x['acceleratorType'], x['state']] for x in r.json()['nodes']]
        return rlist

    def delete(self, tpu_name, sync=False):
        url = f'{self.base_url}/v1alpha1/projects/{self.project}/locations/{self.zone}/nodes/{tpu_name}'

        r = requests.delete(url, headers=self.headers)
        self._raise_for_status(r)
        # print(f'Deleting {tpu_name}...')
        if sync:
            self._wait(tpu_name, 'NOT_FOUND')

    def get(self, tpu_name):
        url = f'{self.base_url}/v1alpha1/projects/{self.project}/locations/{self.zone}/nodes/{tpu_name}'

        r = requests.get(url, headers=self.headers)
        if r.reason == 'Not Found':
            # tpu node not found
            return {'name': tpu_name, 'state': 'NOT_FOUND', 'acceleratorType': None, 'tensorflowVersion': None}

        self._raise_for_status(r)
        return r.json()

    def _wait(self, tpu_name, wait_state=None):

        if wait_state is None:
            wait_state = ['READY', 'STOPPED', 'PREEMPTED', 'TERMINATED', 'NOT_FOUND']
        else:
            wait_state = [wait_state]

        for _ in range(180):
            node_info = self.get(tpu_name)

            # https://cloud.google.com/tpu/docs/reference/rest/v1alpha1/projects.locations.nodes#State
            if node_info['state'] in wait_state:
                return node_info

            time.sleep(3)

        raise TimeoutError(f'TPU node {tpu_name} timed out')

    @staticmethod
    def _raise_for_status(r):
        if 'error' in r.json():
            raise BaseException(r.json()['error'])

    @staticmethod
    def create_graceful(name, type):
        manager = TPUManager()
        node = manager.get(name)
        print(f'TPU Node: {name}')
        print(f'\tstatus: {node["state"]}')
        print(f'\ttype: {node["acceleratorType"]}')
        print(f'\ttf version: {node["tensorflowVersion"]}')

        if node['state'] in {'STOPPED', 'PREEMPTED'}:
            print(f'delete TPU node: {name}')
            manager.delete(tpu_name=name, sync=True)
        if node['state'] in {'READY'}:
            if (type is not None and node['acceleratorType'] != type) or node['tensorflowVersion'] != manager.version():
                print(f'update TPU node: {name}')
                print(f'\ttype:{node["acceleratorType"]} != {type}')
                print(f'\ttf version:{node["tensorflowVersion"]} != {manager.version()}')
                manager.delete(tpu_name=name, sync=True)
        if manager.get(name)['state'] in {'NOT_FOUND'}:
            if type is None:
                print(f'must be a set type')
                assert type is not None
            print(f'creating TPU node: {name}({type})')
            manager.create(tpu_type=type, tpu_name=name)

def create(conf):
    # https://github.com/google/automl/blob/2dbfb7984c20c24d856b96b54498799ed1b270cb/efficientdet/keras/eval.py#L56

    if conf.mode == 'tpu':
        TPUManager(conf.tpu_project).create_graceful(conf.tpu_name, conf.tpu_type)
        tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=conf.tpu_name,)
        tf.config.experimental_connect_to_cluster(tpu_cluster_resolver)
        tf.tpu.experimental.initialize_tpu_system(tpu_cluster_resolver)
        ds_strategy = tf.distribute.TPUStrategy(tpu_cluster_resolver)
        
        policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
        tf.keras.mixed_precision.experimental.set_policy(policy)

        logging.info('All devices: %s', tf.config.list_logical_devices('TPU'))
    elif conf.mode == 'gpus':
        ds_strategy = tf.distribute.MirroredStrategy()

        if conf.mixed_precision == True:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)

        logging.info('All devices: %s', tf.config.list_physical_devices('GPU'))
    else:
        if tf.config.list_physical_devices('GPU'):
            ds_strategy = tf.distribute.OneDeviceStrategy('device:GPU:0')
            
            if conf.mixed_precision == True:
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
        else:
            ds_strategy = tf.distribute.OneDeviceStrategy('device:CPU:0')
    
    return ds_strategy