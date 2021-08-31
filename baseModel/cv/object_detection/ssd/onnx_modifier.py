import argparse
from ast import increment_lineno
import onnx
import onnx_graphsurgeon as gs
import numpy as np

def convert_nms(graph):
    # TODO convert all nms nodes
    nms = None
    for node in graph.nodes:
        if node.op == 'NonMaxSuppression':
            nms = node
    if nms is None:
        print("warn: no NMS is found in the graph")
        return graph
    
    boxes = nms.inputs[0]
    
    transpose = None
    scores_ori = nms.inputs[1]
    for node in graph.nodes:
        if node.op =='Transpose' and scores_ori in node.outputs:
            transpose = node
    score_pretrans_tensor = transpose.inputs[0]

    if transpose is None:
        # TODO add a transpose if no transpose is found
        print("no transpose for scores for NMS is found")

    unsqueeze_out = gs.Variable(name='boxes_unsqueezed_NMS_in', dtype=np.float32)
    unsqueeze_node = gs.Node(op='Unsqueeze',inputs=[boxes], outputs=[unsqueeze_out], attrs={"axes":[2]})
    graph.nodes.append(unsqueeze_node)

    keepTopK = 200
    #batch_size = 1
    # TODO, modify shape[0] for dynamic or static
    num_detections = gs.Variable(name='num_detection',dtype=np.int32,shape=(-1,1))
    nmsed_boxes = gs.Variable(name='nmsed_boxes',dtype=np.float32,shape=(-1,keepTopK,4))
    nmsed_scores = gs.Variable(name='nmsed_scores',dtype=np.float32,shape=(-1,keepTopK))
    nmsed_classed = gs.Variable(name='nmsed_classed',dtype=np.float32,shape=(-1,keepTopK))
    numClasses = 80 # TODO
    topK = 3500 #TODO, get dynamic by nms_pre = cfg.get('deploy_nms_pre', 1000) or 
    scoreThreshold = 0.02 #TODO, get dynamic
    iouThreshold = 0.45 #TODO, get dynamic

    nms_trt = gs.Node(op='BatchedNMSDynamic_TRT', inputs=[unsqueeze_out,score_pretrans_tensor],
            outputs=[num_detections,nmsed_boxes,nmsed_scores,nmsed_classed],
            attrs={"shareLocation":True,
            "backgroundLabelId":-1,
            'numClasses':numClasses,
            'topK':topK,
            'keepTopK':keepTopK,
            'scoreThreshold':scoreThreshold,
            'iouThreshold':iouThreshold,
            'isNormalized':False,
            'clipBoxes':False,
            'scoreBits':16}
            )

    graph.nodes.append(nms_trt)

    graph.outputs = [num_detections,nmsed_boxes,nmsed_scores,nmsed_classed]

    graph.cleanup()
    for node in graph.nodes:
        if node.op =='TopK':
            indix= gs.Variable(name=node.name+'_out1',dtype=np.float32)
            node.outputs.insert(0,indix)
    print(graph)

def test_convert_nms():
    # SSD-512
    inmodel = '/home/jliu/data/models/ssd512_coco_shape512x512_orinmsorigather_topk3500_dynamicbatch_mmdet2.12.onnx'
    outmodel ='/home/jliu/data/models/ssd512_coco_shape512x512_orinmsorigather_topk3500_dynamicbatch_mmdet2.12_gs.onnx'
    # SSD-mobilenetV2
    inmodel = '/home/jliu/data/models/ssd_mobilenetv2_kwj_shape512x512_orinmsorigather_topk1000_dynamicbatch_mmdet2.12.onnx'
    outmodel ='/home/jliu/data/models/ssd_mobilenetv2_kwj_shape512x512_orinmsorigather_topk1000_dynamicbatch_mmdet2.12_gs.onnx'
    graph = gs.import_onnx(onnx.load(inmodel))
    convert_nms(graph)
    onnx.save(gs.export_onnx(graph), outmodel)

def modify_onnx(in_model_or_path, out_model_path=None, modify_nms=True):
    '''
    in_model_or_path: input onnx model or it's path
    out_model_path: output model path, if None, then the output is model
    output: return output model or it's path
    '''
    if isinstance(in_model_or_path, str):
        in_model = onnx.load(in_model_or_path)
    else:
        in_model = in_model_or_path
    graph = gs.import_onnx(in_model)
    if modify_nms:
        convert_nms(graph)
    out_model = gs.export_onnx(graph)
    if out_model_path is not None:
        onnx.save(out_model, out_model_path)
        return out_model_path
    else:
        return out_model

if __name__ == '__main__':
    # test_convert_nms()
    parser = argparse.ArgumentParser(
    description='Convert ONNX')
    parser.add_argument('onnxin', help='test config file path')
    parser.add_argument('onnxout', help='checkpoint file')
    args = parser.parse_args()

    modify_onnx(args.onnxin, args.onnxout)