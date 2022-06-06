import json

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

def delete_endpoint(client, endpoint_name):
    response = client.describe_endpoint_config(EndpointConfigName=endpoint_name)
    model_name = response['ProductionVariants'][0]['ModelName']

    client.delete_model(ModelName=model_name)    
    client.delete_endpoint(EndpointName=endpoint_name)
    client.delete_endpoint_config(EndpointConfigName=endpoint_name)    
    
    print(f'--- Deleted model: {model_name}')
    print(f'--- Deleted endpoint: {endpoint_name}')
    print(f'--- Deleted endpoint_config: {endpoint_name}')    

