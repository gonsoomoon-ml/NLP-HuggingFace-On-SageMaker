import json
import numpy as np

def invoke_endpoint_hf(runtime_client, endpoint_name, payload, content_type):
    '''
    로컬 엔드 포인트 호출
    '''
    response = runtime_client.invoke_endpoint(
        EndpointName=endpoint_name, 
        ContentType=content_type, 
        # Accept='application/json',
        Body=payload,
        )

    result = response['Body'].read().decode().splitlines()   
    result = result[0] # 리스트의 첫번째 항목
    
    result = json.loads(result)
    
    return result


def invoke_endpoint(runtime_client, endpoint_name, payload, content_type):
    '''
    로컬 엔드 포인트 호출
    '''
    response = runtime_client.invoke_endpoint(
        EndpointName=endpoint_name, 
        ContentType=content_type, 
        # Accept='application/json',
        Body=payload,
        )

    result = response['Body'].read().decode().splitlines()    
    
    return result

def inference_batch(df, chunk_size, runtime_client, hf_predictor):
    '''
    배치 사이즈씩 추론 함.
    '''
    def chunker(seq, size):
        '''
        chunk 만큼 데이터 제공
        '''
        return (seq[pos:pos + size] for pos in range(0, len(seq), size))
    
    pred_list = []
    score_list = []

    for rec in chunker(df, chunk_size):
    #     print((rec.doc.tolist()))
    #     print((rec.label.tolist()))    
        doc = rec.doc.tolist()
        payload = {
           "inputs": doc
        }
        payload_dump = json.dumps(payload)
        result = invoke_endpoint_hf(runtime_client, hf_predictor.endpoint_name, 
                             payload_dump,
                             content_type='application/json'
                            )


        for ele in result:
            #print(ele)
            pred = ele["label"]
            pred = int(pred.split('_')[-1]) # 레이블 추출
            score = round(ele["score"],3)   # 스코어 추출

            pred_list.append(pred)
            score_list.append(score)

    return pred_list, score_list



def delete_endpoint(client, endpoint_name):
    response = client.describe_endpoint_config(EndpointConfigName=endpoint_name)
    model_name = response['ProductionVariants'][0]['ModelName']

    client.delete_model(ModelName=model_name)    
    client.delete_endpoint(EndpointName=endpoint_name)
    client.delete_endpoint_config(EndpointConfigName=endpoint_name)    
    
    print(f'--- Deleted model: {model_name}')
    print(f'--- Deleted endpoint: {endpoint_name}')
    print(f'--- Deleted endpoint_config: {endpoint_name}')    


def plot_confusion_matrix(cm, target_names=None, cmap=None, normalize=True, labels=True, title='Confusion matrix'):
    import itertools
    import matplotlib.pyplot as plt
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names)
        plt.yticks(tick_marks, target_names)
    
    if labels:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()    