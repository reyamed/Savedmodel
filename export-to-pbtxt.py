import tensorflow as tf
optimized_graph_path = "./optmized_graph_final.pb"
output_pbtxt = "./optmized_graph_final.pbtxt"# Read the graph.
with tf.gfile.FastGFile(optimized_graph_path, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())# Remove Const nodes.
for i in reversed(range(len(graph_def.node))):
    if graph_def.node[i].op == 'Const':
        del graph_def.node[i]
    for attr in ['T', 'data_format', 'Tshape', 'N', 'Tidx', 'Tdim',
                 'use_cudnn_on_gpu', 'Index', 'Tperm', 'is_training',
                 'Tpaddings']:
        if attr in graph_def.node[i].attr:
            del graph_def.node[i].attr[attr]# Save as text.
tf.train.write_graph(graph_def, "", output_pbtxt, as_text=True)