?	y??7@y??7@!y??7@	2??>?a@2??>?a@!2??>?a@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6y??7@??tps@11DN_??'@A??d????I'?_[o@Y?????*	??|?r?@2?
LIterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::ParallelMapV22??)?D?@!?tP&S@)??)?D?@1?tP&S@:Preprocessing2?
YIterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::ParallelMapV2::TensorSlice2?U???F??!???5@)?U???F??1???5@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch??? ???!P?/e&u??)??? ???1P?/e&u??:Preprocessing2t
=Iterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch$??\????![???[???)$??\????1[???[???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism߈?Y?h??!???????)2!撪???16?s??X??:Preprocessing2F
Iterator::Model	?????!.H?????)? -??1?2G????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 23.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"?22.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no93??>?a@I?1	???F@Q??v8?I@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??tps@??tps@!??tps@      ??!       "	1DN_??'@1DN_??'@!1DN_??'@*      ??!       2	??d??????d????!??d????:	'?_[o@'?_[o@!'?_[o@B      ??!       J	??????????!?????R      ??!       Z	??????????!?????b      ??!       JGPUY3??>?a@b q?1	???F@y??v8?I@?"x
Mgradient_tape/CapsNetMod/Encoder/primary_caps/conv/Conv2D/Conv2DBackpropInputConv2DBackpropInput?s?ihؼ?!?s?ihؼ?0"Y
;gradient_tape/CapsNetMod/Encoder/digit_caps/einsum/Einsum_1Einsumy??ߚ??!??+?9??0"o
Dgradient_tape/CapsNetMod/Encoder/conv2d_3/Conv2D/Conv2DBackpropInputConv2DBackpropInput@h??????!LH@Q????0"@
"CapsNetMod/Encoder/conv2d_3/Conv2DConv2D??G~S??!??>$??0"q
Egradient_tape/CapsNetMod/Encoder/conv2d_3/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?F:is???!??e??7??0"q
Egradient_tape/CapsNetMod/Encoder/conv2d_2/Conv2D/Conv2DBackpropFilterConv2DBackpropFiltero+AF}??!?V?O2G??0"o
Dgradient_tape/CapsNetMod/Encoder/conv2d_2/Conv2D/Conv2DBackpropInputConv2DBackpropInput?Z??[??!""é???0"@
"CapsNetMod/Encoder/conv2d_2/Conv2DConv2D?Xq?ܩ??!6MqG?g??0"q
Egradient_tape/CapsNetMod/Encoder/conv2d_1/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??9????!|?T'd???0"I
+CapsNetMod/Encoder/digit_caps/einsum/EinsumEinsum??k?{??!u?^???0Q      Y@Y\?? M@a???G?D@q?.B??@y???%?W??"?
both?Your program is POTENTIALLY input-bound because 23.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?22.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?31.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 