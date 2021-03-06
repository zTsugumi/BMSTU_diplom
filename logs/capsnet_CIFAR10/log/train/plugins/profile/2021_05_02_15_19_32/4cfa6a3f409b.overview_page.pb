?	?N??@?N??@!?N??@      ??!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-?N??@{????J@1??Z@A???=?>??I~?,|@*	7?A`%??@2g
0Iterator::Model::MaxIntraOpParallelism::Prefetchge????@!j?v??W@)ge????@1j?v??W@:Preprocessing2}
FIterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::Shuffle2?n?o?>??!o??O3O@)?n?o?>??1o??O3O@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism	PS???@!?۷g??W@)????·??1(??g?!??:Preprocessing2t
=Iterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch?lscz¢?!???r???)?lscz¢?1???r???:Preprocessing2F
Iterator::Model-$`ty@!?.?#SX@)?#?????1?vS????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI?*?b??QU??w?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	{????J@{????J@!{????J@      ??!       "	??Z@??Z@!??Z@*      ??!       2	???=?>?????=?>??!???=?>??:	~?,|@~?,|@!~?,|@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q?*?b??yU??w?X@?"V
8gradient_tape/CapsNet/Encoder/digit_caps/einsum/Einsum_1Einsum?~?i????!?~?i????0"F
(CapsNet/Encoder/digit_caps/einsum/EinsumEinsum????2	??!;???qi??0"w
Kgradient_tape/CapsNet/Encoder/primary_caps/conv/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter/??Hz???!??`~??0"F
(CapsNet/Encoder/primary_caps/conv/Conv2DConv2DW?ŭ?!sأ&?Z??0"u
Jgradient_tape/CapsNet/Encoder/primary_caps/conv/Conv2D/Conv2DBackpropInputConv2DBackpropInputH"?????!??5e????0"N
5gradient_tape/CapsNet/Encoder/digit_caps/einsum/Sum_1SumV?RP?;??!???????"T
6gradient_tape/CapsNet/Encoder/digit_caps/einsum/EinsumEinsum=??s????!zRX??o??0"0
Adam/gradients/AddN_12AddN
?????!?? ?L???"K
2gradient_tape/CapsNet/Encoder/digit_caps/mul_3/SumSum?~ه?k??!???E???"9
 CapsNet/Encoder/digit_caps/Sum_2Sum???<k??!$?R7?[??Q      Y@Y1@a??????T@q?=?9`?#@yC?$o?R?"?	
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 