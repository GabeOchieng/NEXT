from jinja2 import Environment, FileSystemLoader
import json
import os
import next.broker.broker
from next.api.resource_manager import ResourceManager
from next.api.targetmapper import TargetMapper

# Use the current directory for widget templates
TEMPLATES_DIRECTORY = os.path.dirname(__file__)
loader = FileSystemLoader(TEMPLATES_DIRECTORY)
env = Environment(loader=loader)

resource_manager = ResourceManager()
broker = next.broker.broker.JobBroker()
targetmapper = TargetMapper() 


class WidgetGenerator():
    
    def getQuery(self, args):
        """
        Generates a getQuery widget. 
        
        Input: ::\n
        	(dict) args 
        
        Output: ::\n
		(str) getQuery widget.
        """
        exp_uid = args['exp_uid']
        app_id = args['app_id']
        if 'participant_uid' in args['args'].keys():
            args['args']['participant_uid'] = '{}_{}'.format(exp_uid,
                                                         args['args']['participant_uid'])
        args_json = json.dumps(args['args'])
        response_json,didSucceed,message = broker.applyAsync(app_id,
                                                             exp_uid,
                                                             'getQuery',
                                                             args_json)        

        response_dict = json.loads(response_json)
        index = response_dict['target_indices'][0]['index']
        query = {}
        query['context'] = response_dict['context']
        query['context_type'] = response_dict['context_type']
        query['target'] = targetmapper.get_target_data(exp_uid, index)
        query['labels'] = response_dict['labels']
        template = env.get_template('getQuery_widget.html')
            
        return {'html': template.render(query = query),
                'args': response_dict }


    
    def processAnswer(self,args):
        """
        Generates a processAnswer widget. Uses the args format as specified in::\n
    		/next_backend/next/learningLibs/apps/TupleBanditsPureExploration
        
        Input: ::\n
        	(dict) args 
        """
        exp_uid = args["exp_uid"]
        app_id = resource_manager.get_app_id(exp_uid)
        
        try:
            target_reward = args['args']['target_reward']
        except:
            return {'message':('Failed to specify all arguments '
                               'or misformed arguments'),
                    'code':400,
                    'status':'FAIL',
                    'base_error':('[target_reward]. Missing required parameter'
                                  'in the JSON body or the post body'
                                  'or the query string')}, 400
        
        # Set the index winner.
        args['args']['target_reward'] = target_reward

        # Args from dict to json type
        args_json = json.dumps(args['args']) 
        # Execute processAnswer 
        response_json,didSucceed,message = broker.applyAsync(app_id,
                                                             exp_uid,
                                                             'processAnswer',
                                                             args_json)
        return { 'html':'success'}


    def getInfo(self,args):
        """
        Generates a getInfo widget. Uses the args format as specified in::\n
            /next_backend/next/learningLibs/apps/TupleBanditsPureExploration
        
        Input: ::\n
            (dict) args 
        """
        info = {}
        response = resource_manager.get_experiment(args['exp_uid'])
        instructions_string = ('Click on bottom target '
                              'that is most similar to the top.')
        info['instructions'] = response.get('instructions',
                                            instructions_string)
        info['debrief'] = response.get('debrief', 'Thanks for participating')
        info['num_tries'] = response.get('num_tries', 100)
        print info
        return {'response': info}
