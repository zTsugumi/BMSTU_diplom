?	m????~@m????~@!m????~@	iޠֲ?iޠֲ?!iޠֲ?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6m????~@Afg?;?@1?+?~@Aam?????Il?f??@Yw?T????*	?E????s@2}
FIterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::Shuffle2??Kǜg??!??ﬄD@)??Kǜg??1??ﬄD@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?i??_=??!?;?E?B@)?4?($???1?|??6@:Preprocessing2t
=Iterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch?i?q????!'?????1@)?i?q????1'?????1@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?ihwH??![?'	t?/@)?ihwH??1[?'	t?/@:Preprocessing2F
Iterator::Model;?O??n??!?!6?g?D@)z??y???1n.??#B
@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9jޠֲ?I???????Q?;MКX@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	Afg?;?@Afg?;?@!Afg?;?@      ??!       "	?+?~@?+?~@!?+?~@*      ??!       2	am?????am?????!am?????:	l?f??@l?f??@!l?f??@B      ??!       J	w?T????w?T????!w?T????R      ??!       Z	w?T????w?T????!w?T????b      ??!       JGPUYjޠֲ?b q???????y?;MКX@?"V
8gradient_tape/CapsNet/Encoder/digit_caps/einsum/Einsum_1Einsum?X??&??!?X??&??0"w
Kgradient_tape/CapsNet/Encoder/primary_caps/conv/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter????P??!F?=??0"F
(CapsNet/Encoder/digit_caps/einsum/EinsumEinsum?0Y?6??!6?ifd??0"F
(CapsNet/Encoder/primary_caps/conv/Conv2DConv2DNWí?;??!??E?!H??0"u
Jgradient_tape/CapsNet/Encoder/primary_caps/conv/Conv2D/Conv2DBackpropInputConv2DBackpropInput???y1??!$?$9???0"N
5gradient_tape/CapsNet/Encoder/digit_caps/einsum/Sum_1Sum?? ????!?g2???"T
6gradient_tape/CapsNet/Encoder/digit_caps/einsum/EinsumEinsum/??A???!2?s?Zd??0"0
Adam/gradients/AddN_12AddNj[5???!??G˩???"K
2gradient_tape/CapsNet/Encoder/digit_caps/mul_3/SumSum	?u?:???!?w??
??"9
 CapsNet/Encoder/digit_caps/Sum_5Sum_??<J???!]"?Q??Q      Y@Y1@a??????T@q?R1F@@y@?8??Q?"?

device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?32.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 