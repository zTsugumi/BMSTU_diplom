?	??,?<W@??,?<W@!??,?<W@	???*@???*@!???*@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6??,?<W@?7?k?g@1??X6s?Q@A#K?X??I?????%@Y?^
?-@*	?rh????@2?
LIterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::ParallelMapV22K %vm{0@!????6?W@)K %vm{0@1????6?W@:Preprocessing2?
YIterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch::ParallelMapV2::TensorSlice2?g\8??!?C?%f?@)?g\8??1?C?%f?@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?0??B???!?:???x??)?0??B???1?:???x??:Preprocessing2t
=Iterator::Model::MaxIntraOpParallelism::Prefetch::MapAndBatch?^?2ᗢ?!3}?	???)?^?2ᗢ?13}?	???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?n????!:-|:???)dX??G??1??V???:Preprocessing2F
Iterator::Model?аu???!㘬????)???ǵ???1??i??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 7.0% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.moderate"?11.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*moderate2s4.2 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9???*@I`??ɣ"0@QZͩ4S@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?7?k?g@?7?k?g@!?7?k?g@      ??!       "	??X6s?Q@??X6s?Q@!??X6s?Q@*      ??!       2	#K?X??#K?X??!#K?X??:	?????%@?????%@!?????%@B      ??!       J	?^
?-@?^
?-@!?^
?-@R      ??!       Z	?^
?-@?^
?-@!?^
?-@b      ??!       JGPUY???*@b q`??ɣ"0@yZͩ4S@?"V
8gradient_tape/CapsNet/Encoder/digit_caps/einsum/Einsum_1EinsumqPT9R???!qPT9R???0"w
Kgradient_tape/CapsNet/Encoder/primary_caps/conv/Conv2D/Conv2DBackpropFilterConv2DBackpropFiltervo<?
a??!'??dʶ??0"u
Jgradient_tape/CapsNet/Encoder/primary_caps/conv/Conv2D/Conv2DBackpropInputConv2DBackpropInput???uU???! ?E?n??0"F
(CapsNet/Encoder/primary_caps/conv/Conv2DConv2DW??????!??<?G??0"F
(CapsNet/Encoder/digit_caps/einsum/EinsumEinsum???????!)?0?k???0"N
5gradient_tape/CapsNet/Encoder/digit_caps/einsum/Sum_1SumSG?s? ??!d??8pE??"T
6gradient_tape/CapsNet/Encoder/digit_caps/einsum/EinsumEinsum?ney???!9??????0"0
Adam/gradients/AddN_12AddN??9?p??!?????H??"=
CapsNet/Encoder/conv2d/Relu_FusedConv2D??P?w???!???\???"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdam??M???!?bN???Q      Y@Y
"P7??L@a?ݯ?fE@qI{_a[1@yo;?E*ts?"?
both?Your program is MODERATELY input-bound because 7.0% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?11.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.moderate"s4.2 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"GPU(: B 