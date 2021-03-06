?	?????p@?????p@!?????p@	j??J?Y??j??J?Y??!j??J?Y??"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6?????p@Ωd ?@1?C ??o@AϽ?K??IYvQ? "@Y??:q9^??*	?x?&1`r@2}
FIterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::Shuffle2"? ˂???!??3?LG@)"? ˂???1??3?LG@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetchw+Kt?Y??!?0d?9?2@)w+Kt?Y??1?0d?9?2@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismfj?!???!=9r?W?A@)T??????1?A??uq0@:Preprocessing2t
=Iterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch??g?ej??!l$?.?w(@)??g?ej??1l$?.?w(@:Preprocessing2F
Iterator::Model@ޫV&???!ѽ? V?D@)h?ej???1?$? ??@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.moderate"?3.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9i??J?Y??I@ȵ(@Qx3Z?ǴW@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	Ωd ?@Ωd ?@!Ωd ?@      ??!       "	?C ??o@?C ??o@!?C ??o@*      ??!       2	Ͻ?K??Ͻ?K??!Ͻ?K??:	YvQ? "@YvQ? "@!YvQ? "@B      ??!       J	??:q9^????:q9^??!??:q9^??R      ??!       Z	??:q9^????:q9^??!??:q9^??b      ??!       JGPUYi??J?Y??b q@ȵ(@yx3Z?ǴW@?"V
8gradient_tape/CapsNet/Encoder/digit_caps/einsum/Einsum_1Einsum?????d??!?????d??0"w
Kgradient_tape/CapsNet/Encoder/primary_caps/conv/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterO???2??!??h?F???0"F
(CapsNet/Encoder/digit_caps/einsum/EinsumEinsum?UIr⡪?!???d5??0"u
Jgradient_tape/CapsNet/Encoder/primary_caps/conv/Conv2D/Conv2DBackpropInputConv2DBackpropInput?Q?4???!??????0"F
(CapsNet/Encoder/primary_caps/conv/Conv2DConv2D?U?2擨?!yv?Y??0"N
5gradient_tape/CapsNet/Encoder/digit_caps/einsum/Sum_1Sum??\!????!?? G???"T
6gradient_tape/CapsNet/Encoder/digit_caps/einsum/EinsumEinsumڡ?u??!??ѿ??0"0
Adam/gradients/AddN_12AddN?8?ω?!??rB'??"M
4gradient_tape/CapsNet/Encoder/digit_caps/mul_6/Mul_1Muli??janz?!??G?[??"M
4gradient_tape/CapsNet/Encoder/digit_caps/mul_5/Mul_1Mulh9??Qz?!;OŐ???Q      Y@Y1@a??????T@q?.?\D@y???D?EZ?"?
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?3.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?40.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 