МЪ
ж
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
,
Floor
x"T
y"T"
Ttype:
2
.
Identity

input"T
output"T"	
Ttype
9
	IdentityN

input2T
output2T"
T
list(type)(0
$

LogicalAnd
x

y

z

q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:

RandomUniformInt

shape"T
minval"Tout
maxval"Tout
output"Tout"
seedint "
seed2int "
Touttype:
2	"
Ttype:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
С
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring Ј
^
StatelessRandomGetKeyCounter
seed"Tseed
key
counter"
Tseedtype0	:
2	

StatelessRandomUniformV2
shape"Tshape
key
counter
alg
output"dtype"
dtypetype0:
2"
Tshapetype0:
2	

StatelessWhile

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint

@
StaticRegexFullMatch	
input

output
"
patternstring
і
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
А
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handleщшelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListReserve
element_shape"
shape_type
num_elements(
handleщшelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsintџџџџџџџџџ
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68Чт
y
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	
*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:
*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0

time_distributed/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nametime_distributed/kernel

+time_distributed/kernel/Read/ReadVariableOpReadVariableOptime_distributed/kernel*!
_output_shapes
:*
dtype0

time_distributed/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nametime_distributed/bias
|
)time_distributed/bias/Read/ReadVariableOpReadVariableOptime_distributed/bias*
_output_shapes	
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
*&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes
:	
*
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:
*
dtype0

Adam/time_distributed/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/time_distributed/kernel/m

2Adam/time_distributed/kernel/m/Read/ReadVariableOpReadVariableOpAdam/time_distributed/kernel/m*!
_output_shapes
:*
dtype0

Adam/time_distributed/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/time_distributed/bias/m

0Adam/time_distributed/bias/m/Read/ReadVariableOpReadVariableOpAdam/time_distributed/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
*&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes
:	
*
dtype0
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
:
*
dtype0

Adam/time_distributed/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/time_distributed/kernel/v

2Adam/time_distributed/kernel/v/Read/ReadVariableOpReadVariableOpAdam/time_distributed/kernel/v*!
_output_shapes
:*
dtype0

Adam/time_distributed/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/time_distributed/bias/v

0Adam/time_distributed/bias/v/Read/ReadVariableOpReadVariableOpAdam/time_distributed/bias/v*
_output_shapes	
:*
dtype0

NoOpNoOp
Д9
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*я8
valueх8Bт8 Bл8
С
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
	optimizer
	variables
trainable_variables
	regularization_losses

	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 

	layer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*

	layer
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses* 

#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses* 
І

)kernel
*bias
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses*

1iter

2beta_1

3beta_2
	4decay
5learning_rate)m*m6m7m)v*v6v7v*
 
60
71
)2
*3*
 
60
71
)2
*3*
* 
А
8non_trainable_variables

9layers
:metrics
;layer_regularization_losses
<layer_metrics
	variables
trainable_variables
	regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

=serving_default* 
* 
* 
* 

>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 
* 
* 
І

6kernel
7bias
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses*

60
71*

60
71*
* 

Inon_trainable_variables

Jlayers
Kmetrics
Llayer_regularization_losses
Mlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
Ј
Ncell
O
state_spec
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses* 
* 
* 
* 

Vnon_trainable_variables

Wlayers
Xmetrics
Ylayer_regularization_losses
Zlayer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses* 
* 
* 
^X
VARIABLE_VALUEdense_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

)0
*1*

)0
*1*
* 

`non_trainable_variables

alayers
bmetrics
clayer_regularization_losses
dlayer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEtime_distributed/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEtime_distributed/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
* 
'
0
1
2
3
4*

e0
f1*
* 
* 
* 
* 
* 
* 
* 
* 

60
71*

60
71*
* 

gnon_trainable_variables

hlayers
imetrics
jlayer_regularization_losses
klayer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses*
* 
* 
* 

0*
* 
* 
* 

l	variables
mtrainable_variables
nregularization_losses
o	keras_api
p__call__
*q&call_and_return_all_conditional_losses* 
* 
* 
* 
* 


rstates
snon_trainable_variables

tlayers
umetrics
vlayer_regularization_losses
wlayer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses* 
* 
* 
* 
	
0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
	xtotal
	ycount
z	variables
{	keras_api*
I
	|total
	}count
~
_fn_kwargs
	variables
	keras_api*
* 
* 
* 
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
l	variables
mtrainable_variables
nregularization_losses
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses* 
* 
* 
* 
* 
	
N0* 
* 
* 
* 
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

x0
y1*

z	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

|0
}1*

	variables*
* 
* 
* 
* 
* 
{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam/time_distributed/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/time_distributed/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam/time_distributed/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUEAdam/time_distributed/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Ў
serving_default_reshape_inputPlaceholder*>
_output_shapes,
*:(џџџџџџџџџџџџџџџџџџ*
dtype0*3
shape*:(џџџџџџџџџџџџџџџџџџ

StatefulPartitionedCallStatefulPartitionedCallserving_default_reshape_inputtime_distributed/kerneltime_distributed/biasdense_1/kerneldense_1/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference_signature_wrapper_4646
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
м
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp+time_distributed/kernel/Read/ReadVariableOp)time_distributed/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp2Adam/time_distributed/kernel/m/Read/ReadVariableOp0Adam/time_distributed/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOp2Adam/time_distributed/kernel/v/Read/ReadVariableOp0Adam/time_distributed/bias/v/Read/ReadVariableOpConst*"
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *&
f!R
__inference__traced_save_5553
Г
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_1/kerneldense_1/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetime_distributed/kerneltime_distributed/biastotalcounttotal_1count_1Adam/dense_1/kernel/mAdam/dense_1/bias/mAdam/time_distributed/kernel/mAdam/time_distributed/bias/mAdam/dense_1/kernel/vAdam/dense_1/bias/vAdam/time_distributed/kernel/vAdam/time_distributed/bias/v*!
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__traced_restore_5626ю
Е
б
"__inference_signature_wrapper_4646
reshape_input
unknown:
	unknown_0:	
	unknown_1:	

	unknown_2:

identityЂStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallreshape_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *(
f#R!
__inference__wrapped_model_3538o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:(џџџџџџџџџџџџџџџџџџ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:m i
>
_output_shapes,
*:(џџџџџџџџџџџџџџџџџџ
'
_user_specified_namereshape_input
ї=
\
=__inference_rnn_layer_call_and_return_conditional_losses_3744

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB T
random_uniform/minConst*
_output_shapes
: *
dtype0*
value	B : X
random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB :џџџџЈ
random_uniformRandomUniformIntrandom_uniform/shape:output:0random_uniform/min:output:0random_uniform/max:output:0*
T0*

Tout0*
_output_shapes
: c
 stateless_random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value
B :
stateless_random_uniform/shapePackstrided_slice:output:0)stateless_random_uniform/shape/1:output:0*
N*
T0*
_output_shapes
:a
stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    a
stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ђ
:stateless_random_uniform/StatelessRandomGetKeyCounter/seedPackrandom_uniform:output:0random_uniform:output:0*
N*
T0*
_output_shapes
:Щ
5stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterCstateless_random_uniform/StatelessRandomGetKeyCounter/seed:output:0*
Tseed0* 
_output_shapes
::w
5stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :о
1stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2'stateless_random_uniform/shape:output:0;stateless_random_uniform/StatelessRandomGetKeyCounter:key:0?stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0>stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*(
_output_shapes
:џџџџџџџџџ
stateless_random_uniform/subSub%stateless_random_uniform/max:output:0%stateless_random_uniform/min:output:0*
T0*
_output_shapes
: Д
stateless_random_uniform/mulMul:stateless_random_uniform/StatelessRandomUniformV2:output:0 stateless_random_uniform/sub:z:0*
T0*(
_output_shapes
:џџџџџџџџџ
stateless_random_uniformAddV2 stateless_random_uniform/mul:z:0%stateless_random_uniform/min:output:0*
T0*(
_output_shapes
:џџџџџџџџџc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ъ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask
'spiking_activation_cell/PartitionedCallPartitionedCallstrided_slice_2:output:0stateless_random_uniform:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_spiking_activation_cell_layer_call_and_return_conditional_losses_3686n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Р
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0stateless_random_uniform:z:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*
T
	2*
_lower_using_switch_merge(*
_num_original_outputs*4
_output_shapes"
 : : : : :џџџџџџџџџ: : * 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_3693*
condR
while_cond_3692*3
output_shapes"
 : : : : :џџџџџџџџџ: : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ь
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџe
IdentityIdentitytranspose_1:y:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџџџџџџџџџџ:] Y
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
м'
№
while_body_5082
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ї
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџ*
element_dtype0f
$while/spiking_activation_cell/Cast/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z l
*while/spiking_activation_cell/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 ZВ
(while/spiking_activation_cell/LogicalAnd
LogicalAnd-while/spiking_activation_cell/Cast/x:output:03while/spiking_activation_cell/LogicalAnd/y:output:0*
_output_shapes
: 
"while/spiking_activation_cell/ReluRelu0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*(
_output_shapes
:џџџџџџџџџh
#while/spiking_activation_cell/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:Л
!while/spiking_activation_cell/mulMul0while/spiking_activation_cell/Relu:activations:0,while/spiking_activation_cell/mul/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
!while/spiking_activation_cell/addAddV2while_placeholder_2%while/spiking_activation_cell/mul:z:0*
T0*(
_output_shapes
:џџџџџџџџџ
#while/spiking_activation_cell/FloorFloor%while/spiking_activation_cell/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџЋ
!while/spiking_activation_cell/subSub%while/spiking_activation_cell/add:z:0'while/spiking_activation_cell/Floor:y:0*
T0*(
_output_shapes
:џџџџџџџџџl
'while/spiking_activation_cell/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:О
%while/spiking_activation_cell/truedivRealDiv'while/spiking_activation_cell/Floor:y:00while/spiking_activation_cell/truediv/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
&while/spiking_activation_cell/IdentityIdentity)while/spiking_activation_cell/truediv:z:0*
T0*(
_output_shapes
:џџџџџџџџџ
(while/spiking_activation_cell/Identity_1Identity%while/spiking_activation_cell/sub:z:0*
T0*(
_output_shapes
:џџџџџџџџџю
'while/spiking_activation_cell/IdentityN	IdentityN)while/spiking_activation_cell/truediv:z:0%while/spiking_activation_cell/sub:z:00while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2*
T
2**
_gradient_op_typeCustomGradient-5107*d
_output_shapesR
P:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџй
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder0while/spiking_activation_cell/IdentityN:output:0*
_output_shapes
: *
element_dtype0:щшвM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: L
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: :щшв
while/Identity_4Identity0while/spiking_activation_cell/IdentityN:output:1*
T0*(
_output_shapes
:џџџџџџџџџ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 : : : : :џџџџџџџџџ: : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
д

J__inference_time_distributed_layer_call_and_return_conditional_losses_4703

inputs9
$dense_matmul_readvariableop_resource:4
%dense_biasadd_readvariableop_resource:	
identityЂdense/BiasAdd/ReadVariableOpЂdense/MatMul/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ @  f
ReshapeReshapeinputsReshape/shape:output:0*
T0*)
_output_shapes
:џџџџџџџџџ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*!
_output_shapes
:*
dtype0
dense/MatMulMatMulReshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshapedense/BiasAdd:output:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџo
IdentityIdentityReshape_1:output:0^NoOp*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: џџџџџџџџџџџџџџџџџџ: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:^ Z
6
_output_shapes$
": џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
О>
^
=__inference_rnn_layer_call_and_return_conditional_losses_5249
inputs_0
identity=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB T
random_uniform/minConst*
_output_shapes
: *
dtype0*
value	B : X
random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB :џџџџЈ
random_uniformRandomUniformIntrandom_uniform/shape:output:0random_uniform/min:output:0random_uniform/max:output:0*
T0*

Tout0*
_output_shapes
: c
 stateless_random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value
B :
stateless_random_uniform/shapePackstrided_slice:output:0)stateless_random_uniform/shape/1:output:0*
N*
T0*
_output_shapes
:a
stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    a
stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ђ
:stateless_random_uniform/StatelessRandomGetKeyCounter/seedPackrandom_uniform:output:0random_uniform:output:0*
N*
T0*
_output_shapes
:Щ
5stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterCstateless_random_uniform/StatelessRandomGetKeyCounter/seed:output:0*
Tseed0* 
_output_shapes
::w
5stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :о
1stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2'stateless_random_uniform/shape:output:0;stateless_random_uniform/StatelessRandomGetKeyCounter:key:0?stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0>stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*(
_output_shapes
:џџџџџџџџџ
stateless_random_uniform/subSub%stateless_random_uniform/max:output:0%stateless_random_uniform/min:output:0*
T0*
_output_shapes
: Д
stateless_random_uniform/mulMul:stateless_random_uniform/StatelessRandomUniformV2:output:0 stateless_random_uniform/sub:z:0*
T0*(
_output_shapes
:џџџџџџџџџ
stateless_random_uniformAddV2 stateless_random_uniform/mul:z:0%stateless_random_uniform/min:output:0*
T0*(
_output_shapes
:џџџџџџџџџc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          y
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ъ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask`
spiking_activation_cell/Cast/xConst*
_output_shapes
: *
dtype0
*
value	B
 Zf
$spiking_activation_cell/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z 
"spiking_activation_cell/LogicalAnd
LogicalAnd'spiking_activation_cell/Cast/x:output:0-spiking_activation_cell/LogicalAnd/y:output:0*
_output_shapes
: q
spiking_activation_cell/ReluRelustrided_slice_2:output:0*
T0*(
_output_shapes
:џџџџџџџџџn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Р
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0stateless_random_uniform:z:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*
T
	2*
_lower_using_switch_merge(*
_num_original_outputs*4
_output_shapes"
 : : : : :џџџџџџџџџ: : * 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_5196*
condR
while_cond_5195*3
output_shapes"
 : : : : :џџџџџџџџџ: : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ь
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџe
IdentityIdentitytranspose_1:y:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџџџџџџџџџџ:_ [
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
ї=
\
=__inference_rnn_layer_call_and_return_conditional_losses_3872

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB T
random_uniform/minConst*
_output_shapes
: *
dtype0*
value	B : X
random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB :џџџџЈ
random_uniformRandomUniformIntrandom_uniform/shape:output:0random_uniform/min:output:0random_uniform/max:output:0*
T0*

Tout0*
_output_shapes
: c
 stateless_random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value
B :
stateless_random_uniform/shapePackstrided_slice:output:0)stateless_random_uniform/shape/1:output:0*
N*
T0*
_output_shapes
:a
stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    a
stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ђ
:stateless_random_uniform/StatelessRandomGetKeyCounter/seedPackrandom_uniform:output:0random_uniform:output:0*
N*
T0*
_output_shapes
:Щ
5stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterCstateless_random_uniform/StatelessRandomGetKeyCounter/seed:output:0*
Tseed0* 
_output_shapes
::w
5stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :о
1stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2'stateless_random_uniform/shape:output:0;stateless_random_uniform/StatelessRandomGetKeyCounter:key:0?stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0>stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*(
_output_shapes
:џџџџџџџџџ
stateless_random_uniform/subSub%stateless_random_uniform/max:output:0%stateless_random_uniform/min:output:0*
T0*
_output_shapes
: Д
stateless_random_uniform/mulMul:stateless_random_uniform/StatelessRandomUniformV2:output:0 stateless_random_uniform/sub:z:0*
T0*(
_output_shapes
:џџџџџџџџџ
stateless_random_uniformAddV2 stateless_random_uniform/mul:z:0%stateless_random_uniform/min:output:0*
T0*(
_output_shapes
:џџџџџџџџџc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ъ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask
'spiking_activation_cell/PartitionedCallPartitionedCallstrided_slice_2:output:0stateless_random_uniform:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_spiking_activation_cell_layer_call_and_return_conditional_losses_3769n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Р
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0stateless_random_uniform:z:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*
T
	2*
_lower_using_switch_merge(*
_num_original_outputs*4
_output_shapes"
 : : : : :џџџџџџџџџ: : * 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_3821*
condR
while_cond_3820*3
output_shapes"
 : : : : :џџџџџџџџџ: : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ь
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџe
IdentityIdentitytranspose_1:y:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџџџџџџџџџџ:] Y
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
К

Q__inference_spiking_activation_cell_layer_call_and_return_conditional_losses_3686

inputs

states

identity_2

identity_3H
Cast/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z N
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 ZX

LogicalAnd
LogicalAndCast/x:output:0LogicalAnd/y:output:0*
_output_shapes
: G
ReluReluinputs*
T0*(
_output_shapes
:џџџџџџџџџJ
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:a
mulMulRelu:activations:0mul/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџP
addAddV2statesmul:z:0*
T0*(
_output_shapes
:џџџџџџџџџJ
FloorFlooradd:z:0*
T0*(
_output_shapes
:џџџџџџџџџQ
subSubadd:z:0	Floor:y:0*
T0*(
_output_shapes
:џџџџџџџџџN
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:d
truedivRealDiv	Floor:y:0truediv/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџT
IdentityIdentitytruediv:z:0*
T0*(
_output_shapes
:џџџџџџџџџR

Identity_1Identitysub:z:0*
T0*(
_output_shapes
:џџџџџџџџџн
	IdentityN	IdentityNtruediv:z:0sub:z:0inputsstates*
T
2**
_gradient_op_typeCustomGradient-3669*d
_output_shapesR
P:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ]

Identity_2IdentityIdentityN:output:0*
T0*(
_output_shapes
:џџџџџџџџџ]

Identity_3IdentityIdentityN:output:1*
T0*(
_output_shapes
:џџџџџџџџџ"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџ:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:PL
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_namestates
Ь

Q__inference_spiking_activation_cell_layer_call_and_return_conditional_losses_5299

inputs
states_0
identity

identity_1H
Cast/xConst*
_output_shapes
: *
dtype0
*
value	B
 ZN
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 ZX

LogicalAnd
LogicalAndCast/x:output:0LogicalAnd/y:output:0*
_output_shapes
: G
ReluReluinputs*
T0*(
_output_shapes
:џџџџџџџџџ[
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџS

Identity_1Identitystates_0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџ:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:RN
(
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
states/0
Б
Ь
!__inference_internal_grad_fn_5429
result_grads_0
result_grads_1
result_grads_2
result_grads_3I
Eones_like_shape_spiking_activation_while_spiking_activation_cell_relu
identity
ones_like/ShapeShapeEones_like_shape_spiking_activation_while_spiking_activation_cell_relu*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?x
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџс
Ggradient_tape/spiking_activation/while/spiking_activation_cell/ReluGradReluGradones_like:output:0Eones_like_shape_spiking_activation_while_spiking_activation_cell_relu*
T0*(
_output_shapes
:џџџџџџџџџЂ
mulMulSgradient_tape/spiking_activation/while/spiking_activation_cell/ReluGrad:backprops:0result_grads_0*
T0*(
_output_shapes
:џџџџџџџџџP
IdentityIdentitymul:z:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*w
_input_shapesf
d:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:X T
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_0:XT
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_1:XT
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_2:XT
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_3:.*
(
_output_shapes
:џџџџџџџџџ

n
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_3885

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Њ

while_cond_3692
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_12
.while_while_cond_3692___redundant_placeholder0
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
": : : : :џџџџџџџџџ: :: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
ш
ь
D__inference_sequential_layer_call_and_return_conditional_losses_4237

inputs*
time_distributed_4222:$
time_distributed_4224:	
dense_1_4231:	

dense_1_4233:

identityЂdense_1/StatefulPartitionedCallЂ*spiking_activation/StatefulPartitionedCallЂ(time_distributed/StatefulPartitionedCallЦ
reshape/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *6
_output_shapes$
": џџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_reshape_layer_call_and_return_conditional_losses_3908Е
(time_distributed/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0time_distributed_4222time_distributed_4224*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_time_distributed_layer_call_and_return_conditional_losses_3612o
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ @  Ђ
time_distributed/ReshapeReshape reshape/PartitionedCall:output:0'time_distributed/Reshape/shape:output:0*
T0*)
_output_shapes
:џџџџџџџџџ
*spiking_activation/StatefulPartitionedCallStatefulPartitionedCall1time_distributed/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_spiking_activation_layer_call_and_return_conditional_losses_4194
(global_average_pooling1d/PartitionedCallPartitionedCall3spiking_activation/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_3885
dense_1/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0dense_1_4231dense_1_4233*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_4058w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
Р
NoOpNoOp ^dense_1/StatefulPartitionedCall+^spiking_activation/StatefulPartitionedCall)^time_distributed/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:(џџџџџџџџџџџџџџџџџџ: : : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2X
*spiking_activation/StatefulPartitionedCall*spiking_activation/StatefulPartitionedCall2T
(time_distributed/StatefulPartitionedCall(time_distributed/StatefulPartitionedCall:f b
>
_output_shapes,
*:(џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Г
Г
!__inference_internal_grad_fn_5485
result_grads_0
result_grads_1
result_grads_2
result_grads_30
,ones_like_shape_spiking_activation_cell_relu
identityk
ones_like/ShapeShape,ones_like_shape_spiking_activation_cell_relu*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?x
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЏ
.gradient_tape/spiking_activation_cell/ReluGradReluGradones_like:output:0,ones_like_shape_spiking_activation_cell_relu*
T0*(
_output_shapes
:џџџџџџџџџ
mulMul:gradient_tape/spiking_activation_cell/ReluGrad:backprops:0result_grads_0*
T0*(
_output_shapes
:џџџџџџџџџP
IdentityIdentitymul:z:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*w
_input_shapesf
d:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:X T
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_0:XT
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_1:XT
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_2:XT
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_3:.*
(
_output_shapes
:џџџџџџџџџ

Ц
!__inference_internal_grad_fn_5415
result_grads_0
result_grads_1
result_grads_2
result_grads_3C
?ones_like_shape_spiking_activation_spiking_activation_cell_relu
identity~
ones_like/ShapeShape?ones_like_shape_spiking_activation_spiking_activation_cell_relu*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?x
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџе
Agradient_tape/spiking_activation/spiking_activation_cell/ReluGradReluGradones_like:output:0?ones_like_shape_spiking_activation_spiking_activation_cell_relu*
T0*(
_output_shapes
:џџџџџџџџџ
mulMulMgradient_tape/spiking_activation/spiking_activation_cell/ReluGrad:backprops:0result_grads_0*
T0*(
_output_shapes
:џџџџџџџџџP
IdentityIdentitymul:z:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*w
_input_shapesf
d:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:X T
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_0:XT
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_1:XT
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_2:XT
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_3:.*
(
_output_shapes
:џџџџџџџџџ
Њ

while_cond_4794
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_12
.while_while_cond_4794___redundant_placeholder0
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
": : : : :џџџџџџџџџ: :: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
м'
№
while_body_3978
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ї
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџ*
element_dtype0f
$while/spiking_activation_cell/Cast/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z l
*while/spiking_activation_cell/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 ZВ
(while/spiking_activation_cell/LogicalAnd
LogicalAnd-while/spiking_activation_cell/Cast/x:output:03while/spiking_activation_cell/LogicalAnd/y:output:0*
_output_shapes
: 
"while/spiking_activation_cell/ReluRelu0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*(
_output_shapes
:џџџџџџџџџh
#while/spiking_activation_cell/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:Л
!while/spiking_activation_cell/mulMul0while/spiking_activation_cell/Relu:activations:0,while/spiking_activation_cell/mul/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
!while/spiking_activation_cell/addAddV2while_placeholder_2%while/spiking_activation_cell/mul:z:0*
T0*(
_output_shapes
:џџџџџџџџџ
#while/spiking_activation_cell/FloorFloor%while/spiking_activation_cell/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџЋ
!while/spiking_activation_cell/subSub%while/spiking_activation_cell/add:z:0'while/spiking_activation_cell/Floor:y:0*
T0*(
_output_shapes
:џџџџџџџџџl
'while/spiking_activation_cell/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:О
%while/spiking_activation_cell/truedivRealDiv'while/spiking_activation_cell/Floor:y:00while/spiking_activation_cell/truediv/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
&while/spiking_activation_cell/IdentityIdentity)while/spiking_activation_cell/truediv:z:0*
T0*(
_output_shapes
:џџџџџџџџџ
(while/spiking_activation_cell/Identity_1Identity%while/spiking_activation_cell/sub:z:0*
T0*(
_output_shapes
:џџџџџџџџџю
'while/spiking_activation_cell/IdentityN	IdentityN)while/spiking_activation_cell/truediv:z:0%while/spiking_activation_cell/sub:z:00while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2*
T
2**
_gradient_op_typeCustomGradient-4003*d
_output_shapesR
P:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџй
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder0while/spiking_activation_cell/IdentityN:output:0*
_output_shapes
: *
element_dtype0:щшвM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: L
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: :щшв
while/Identity_4Identity0while/spiking_activation_cell/IdentityN:output:1*
T0*(
_output_shapes
:џџџџџџџџџ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 : : : : :џџџџџџџџџ: : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
м'
№
while_body_4795
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ї
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџ*
element_dtype0f
$while/spiking_activation_cell/Cast/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z l
*while/spiking_activation_cell/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 ZВ
(while/spiking_activation_cell/LogicalAnd
LogicalAnd-while/spiking_activation_cell/Cast/x:output:03while/spiking_activation_cell/LogicalAnd/y:output:0*
_output_shapes
: 
"while/spiking_activation_cell/ReluRelu0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*(
_output_shapes
:џџџџџџџџџh
#while/spiking_activation_cell/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:Л
!while/spiking_activation_cell/mulMul0while/spiking_activation_cell/Relu:activations:0,while/spiking_activation_cell/mul/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
!while/spiking_activation_cell/addAddV2while_placeholder_2%while/spiking_activation_cell/mul:z:0*
T0*(
_output_shapes
:џџџџџџџџџ
#while/spiking_activation_cell/FloorFloor%while/spiking_activation_cell/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџЋ
!while/spiking_activation_cell/subSub%while/spiking_activation_cell/add:z:0'while/spiking_activation_cell/Floor:y:0*
T0*(
_output_shapes
:џџџџџџџџџl
'while/spiking_activation_cell/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:О
%while/spiking_activation_cell/truedivRealDiv'while/spiking_activation_cell/Floor:y:00while/spiking_activation_cell/truediv/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
&while/spiking_activation_cell/IdentityIdentity)while/spiking_activation_cell/truediv:z:0*
T0*(
_output_shapes
:џџџџџџџџџ
(while/spiking_activation_cell/Identity_1Identity%while/spiking_activation_cell/sub:z:0*
T0*(
_output_shapes
:џџџџџџџџџю
'while/spiking_activation_cell/IdentityN	IdentityN)while/spiking_activation_cell/truediv:z:0%while/spiking_activation_cell/sub:z:00while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2*
T
2**
_gradient_op_typeCustomGradient-4820*d
_output_shapesR
P:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџй
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder0while/spiking_activation_cell/IdentityN:output:0*
_output_shapes
: *
element_dtype0:щшвM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: L
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: :щшв
while/Identity_4Identity0while/spiking_activation_cell/IdentityN:output:1*
T0*(
_output_shapes
:џџџџџџџџџ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 : : : : :џџџџџџџџџ: : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
Ъ
б
!__inference_internal_grad_fn_5359
result_grads_0
result_grads_1
result_grads_2
result_grads_3N
Jones_like_shape_sequential_spiking_activation_spiking_activation_cell_relu
identity
ones_like/ShapeShapeJones_like_shape_sequential_spiking_activation_spiking_activation_cell_relu*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?x
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџы
Lgradient_tape/sequential/spiking_activation/spiking_activation_cell/ReluGradReluGradones_like:output:0Jones_like_shape_sequential_spiking_activation_spiking_activation_cell_relu*
T0*(
_output_shapes
:џџџџџџџџџЇ
mulMulXgradient_tape/sequential/spiking_activation/spiking_activation_cell/ReluGrad:backprops:0result_grads_0*
T0*(
_output_shapes
:џџџџџџџџџP
IdentityIdentitymul:z:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*w
_input_shapesf
d:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:X T
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_0:XT
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_1:XT
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_2:XT
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_3:.*
(
_output_shapes
:џџџџџџџџџ

 
/__inference_time_distributed_layer_call_fn_4673

inputs
unknown:
	unknown_0:	
identityЂStatefulPartitionedCall№
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_time_distributed_layer_call_and_return_conditional_losses_3573}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: џџџџџџџџџџџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
6
_output_shapes$
": џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs

n
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_4973

inputs
identityX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
д

J__inference_time_distributed_layer_call_and_return_conditional_losses_4724

inputs9
$dense_matmul_readvariableop_resource:4
%dense_biasadd_readvariableop_resource:	
identityЂdense/BiasAdd/ReadVariableOpЂdense/MatMul/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ @  f
ReshapeReshapeinputsReshape/shape:output:0*
T0*)
_output_shapes
:џџџџџџџџџ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*!
_output_shapes
:*
dtype0
dense/MatMulMatMulReshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshapedense/BiasAdd:output:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџo
IdentityIdentityReshape_1:output:0^NoOp*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: џџџџџџџџџџџџџџџџџџ: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:^ Z
6
_output_shapes$
": џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ш1
ќ
__inference__traced_save_5553
file_prefix-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop6
2savev2_time_distributed_kernel_read_readvariableop4
0savev2_time_distributed_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop=
9savev2_adam_time_distributed_kernel_m_read_readvariableop;
7savev2_adam_time_distributed_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop=
9savev2_adam_time_distributed_kernel_v_read_readvariableop;
7savev2_adam_time_distributed_bias_v_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: с

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*

value
B§	B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value6B4B B B B B B B B B B B B B B B B B B B B B B 	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop2savev2_time_distributed_kernel_read_readvariableop0savev2_time_distributed_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop9savev2_adam_time_distributed_kernel_m_read_readvariableop7savev2_adam_time_distributed_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop9savev2_adam_time_distributed_kernel_v_read_readvariableop7savev2_adam_time_distributed_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *$
dtypes
2	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*
_input_shapes
: :	
:
: : : : : ::: : : : :	
:
:::	
:
::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	
: 

_output_shapes
:
:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :'#
!
_output_shapes
::!	

_output_shapes	
::


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	
: 

_output_shapes
:
:'#
!
_output_shapes
::!

_output_shapes	
::%!

_output_shapes
:	
: 

_output_shapes
:
:'#
!
_output_shapes
::!

_output_shapes	
::

_output_shapes
: 
ш
ь
D__inference_sequential_layer_call_and_return_conditional_losses_4065

inputs*
time_distributed_3910:$
time_distributed_3912:	
dense_1_4059:	

dense_1_4061:

identityЂdense_1/StatefulPartitionedCallЂ*spiking_activation/StatefulPartitionedCallЂ(time_distributed/StatefulPartitionedCallЦ
reshape/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *6
_output_shapes$
": џџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_reshape_layer_call_and_return_conditional_losses_3908Е
(time_distributed/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0time_distributed_3910time_distributed_3912*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_time_distributed_layer_call_and_return_conditional_losses_3573o
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ @  Ђ
time_distributed/ReshapeReshape reshape/PartitionedCall:output:0'time_distributed/Reshape/shape:output:0*
T0*)
_output_shapes
:џџџџџџџџџ
*spiking_activation/StatefulPartitionedCallStatefulPartitionedCall1time_distributed/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_spiking_activation_layer_call_and_return_conditional_losses_4045
(global_average_pooling1d/PartitionedCallPartitionedCall3spiking_activation/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_3885
dense_1/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0dense_1_4059dense_1_4061*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_4058w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
Р
NoOpNoOp ^dense_1/StatefulPartitionedCall+^spiking_activation/StatefulPartitionedCall)^time_distributed/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:(џџџџџџџџџџџџџџџџџџ: : : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2X
*spiking_activation/StatefulPartitionedCall*spiking_activation/StatefulPartitionedCall2T
(time_distributed/StatefulPartitionedCall(time_distributed/StatefulPartitionedCall:f b
>
_output_shapes,
*:(џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
б	
є
?__inference_dense_layer_call_and_return_conditional_losses_5011

inputs3
matmul_readvariableop_resource:.
biasadd_readvariableop_resource:	
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpw
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Й


!__inference_internal_grad_fn_5471
result_grads_0
result_grads_1
result_grads_2
result_grads_3
ones_like_shape_relu
identityS
ones_like/ShapeShapeones_like_shape_relu*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?x
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
gradient_tape/ReluGradReluGradones_like:output:0ones_like_shape_relu*
T0*(
_output_shapes
:џџџџџџџџџq
mulMul"gradient_tape/ReluGrad:backprops:0result_grads_0*
T0*(
_output_shapes
:џџџџџџџџџP
IdentityIdentitymul:z:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*w
_input_shapesf
d:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:X T
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_0:XT
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_1:XT
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_2:XT
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_3:.*
(
_output_shapes
:џџџџџџџџџ
б
Й
!__inference_internal_grad_fn_5499
result_grads_0
result_grads_1
result_grads_2
result_grads_36
2ones_like_shape_while_spiking_activation_cell_relu
identityq
ones_like/ShapeShape2ones_like_shape_while_spiking_activation_cell_relu*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?x
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЛ
4gradient_tape/while/spiking_activation_cell/ReluGradReluGradones_like:output:02ones_like_shape_while_spiking_activation_cell_relu*
T0*(
_output_shapes
:џџџџџџџџџ
mulMul@gradient_tape/while/spiking_activation_cell/ReluGrad:backprops:0result_grads_0*
T0*(
_output_shapes
:џџџџџџџџџP
IdentityIdentitymul:z:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*w
_input_shapesf
d:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:X T
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_0:XT
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_1:XT
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_2:XT
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_3:.*
(
_output_shapes
:џџџџџџџџџ
Ђ
]
"__inference_rnn_layer_call_fn_5021
inputs_0
identityЂStatefulPartitionedCallЫ
StatefulPartitionedCallStatefulPartitionedCallinputs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *F
fAR?
=__inference_rnn_layer_call_and_return_conditional_losses_3872}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџџџџџџџџџџ22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
Т

Q__inference_spiking_activation_cell_layer_call_and_return_conditional_losses_5289

inputs
states_0

identity_2

identity_3H
Cast/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z N
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 ZX

LogicalAnd
LogicalAndCast/x:output:0LogicalAnd/y:output:0*
_output_shapes
: G
ReluReluinputs*
T0*(
_output_shapes
:џџџџџџџџџJ
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:a
mulMulRelu:activations:0mul/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџR
addAddV2states_0mul:z:0*
T0*(
_output_shapes
:џџџџџџџџџJ
FloorFlooradd:z:0*
T0*(
_output_shapes
:џџџџџџџџџQ
subSubadd:z:0	Floor:y:0*
T0*(
_output_shapes
:џџџџџџџџџN
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:d
truedivRealDiv	Floor:y:0truediv/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџT
IdentityIdentitytruediv:z:0*
T0*(
_output_shapes
:џџџџџџџџџR

Identity_1Identitysub:z:0*
T0*(
_output_shapes
:џџџџџџџџџп
	IdentityN	IdentityNtruediv:z:0sub:z:0inputsstates_0*
T
2**
_gradient_op_typeCustomGradient-5272*d
_output_shapesR
P:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ]

Identity_2IdentityIdentityN:output:0*
T0*(
_output_shapes
:џџџџџџџџџ]

Identity_3IdentityIdentityN:output:1*
T0*(
_output_shapes
:џџџџџџџџџ"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџ:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:RN
(
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
states/0
џ
№
while_body_5196
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ї
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџ*
element_dtype0f
$while/spiking_activation_cell/Cast/xConst*
_output_shapes
: *
dtype0
*
value	B
 Zl
*while/spiking_activation_cell/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 ZВ
(while/spiking_activation_cell/LogicalAnd
LogicalAnd-while/spiking_activation_cell/Cast/x:output:03while/spiking_activation_cell/LogicalAnd/y:output:0*
_output_shapes
: 
"while/spiking_activation_cell/ReluRelu0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*(
_output_shapes
:џџџџџџџџџй
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder0while/spiking_activation_cell/Relu:activations:0*
_output_shapes
: *
element_dtype0:щшвM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: L
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: :щшвd
while/Identity_4Identitywhile_placeholder_2*
T0*(
_output_shapes
:џџџџџџџџџ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 : : : : :џџџџџџџџџ: : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
Ђ
]
"__inference_rnn_layer_call_fn_5016
inputs_0
identityЂStatefulPartitionedCallЫ
StatefulPartitionedCallStatefulPartitionedCallinputs_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *F
fAR?
=__inference_rnn_layer_call_and_return_conditional_losses_3744}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџџџџџџџџџџ22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
Ї%
џ
"spiking_activation_while_body_4570B
>spiking_activation_while_spiking_activation_while_loop_counterH
Dspiking_activation_while_spiking_activation_while_maximum_iterations(
$spiking_activation_while_placeholder*
&spiking_activation_while_placeholder_1*
&spiking_activation_while_placeholder_2A
=spiking_activation_while_spiking_activation_strided_slice_1_0}
yspiking_activation_while_tensorarrayv2read_tensorlistgetitem_spiking_activation_tensorarrayunstack_tensorlistfromtensor_0%
!spiking_activation_while_identity'
#spiking_activation_while_identity_1'
#spiking_activation_while_identity_2'
#spiking_activation_while_identity_3'
#spiking_activation_while_identity_4?
;spiking_activation_while_spiking_activation_strided_slice_1{
wspiking_activation_while_tensorarrayv2read_tensorlistgetitem_spiking_activation_tensorarrayunstack_tensorlistfromtensor
Jspiking_activation/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
<spiking_activation/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemyspiking_activation_while_tensorarrayv2read_tensorlistgetitem_spiking_activation_tensorarrayunstack_tensorlistfromtensor_0$spiking_activation_while_placeholderSspiking_activation/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџ*
element_dtype0y
7spiking_activation/while/spiking_activation_cell/Cast/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z
=spiking_activation/while/spiking_activation_cell/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Zы
;spiking_activation/while/spiking_activation_cell/LogicalAnd
LogicalAnd@spiking_activation/while/spiking_activation_cell/Cast/x:output:0Fspiking_activation/while/spiking_activation_cell/LogicalAnd/y:output:0*
_output_shapes
: Е
5spiking_activation/while/spiking_activation_cell/ReluReluCspiking_activation/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*(
_output_shapes
:џџџџџџџџџЅ
=spiking_activation/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem&spiking_activation_while_placeholder_1$spiking_activation_while_placeholderCspiking_activation/while/spiking_activation_cell/Relu:activations:0*
_output_shapes
: *
element_dtype0:щшв`
spiking_activation/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
spiking_activation/while/addAddV2$spiking_activation_while_placeholder'spiking_activation/while/add/y:output:0*
T0*
_output_shapes
: b
 spiking_activation/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Г
spiking_activation/while/add_1AddV2>spiking_activation_while_spiking_activation_while_loop_counter)spiking_activation/while/add_1/y:output:0*
T0*
_output_shapes
: r
!spiking_activation/while/IdentityIdentity"spiking_activation/while/add_1:z:0*
T0*
_output_shapes
: 
#spiking_activation/while/Identity_1IdentityDspiking_activation_while_spiking_activation_while_maximum_iterations*
T0*
_output_shapes
: r
#spiking_activation/while/Identity_2Identity spiking_activation/while/add:z:0*
T0*
_output_shapes
: В
#spiking_activation/while/Identity_3IdentityMspiking_activation/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: :щшв
#spiking_activation/while/Identity_4Identity&spiking_activation_while_placeholder_2*
T0*(
_output_shapes
:џџџџџџџџџ"O
!spiking_activation_while_identity*spiking_activation/while/Identity:output:0"S
#spiking_activation_while_identity_1,spiking_activation/while/Identity_1:output:0"S
#spiking_activation_while_identity_2,spiking_activation/while/Identity_2:output:0"S
#spiking_activation_while_identity_3,spiking_activation/while/Identity_3:output:0"S
#spiking_activation_while_identity_4,spiking_activation/while/Identity_4:output:0"|
;spiking_activation_while_spiking_activation_strided_slice_1=spiking_activation_while_spiking_activation_strided_slice_1_0"є
wspiking_activation_while_tensorarrayv2read_tensorlistgetitem_spiking_activation_tensorarrayunstack_tensorlistfromtensoryspiking_activation_while_tensorarrayv2read_tensorlistgetitem_spiking_activation_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 : : : : :џџџџџџџџџ: : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
Ђ
]
A__inference_reshape_layer_call_and_return_conditional_losses_4664

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџS
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
valueB	 :
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:s
ReshapeReshapeinputsReshape/shape:output:0*
T0*6
_output_shapes$
": џџџџџџџџџџџџџџџџџџg
IdentityIdentityReshape:output:0*
T0*6
_output_shapes$
": џџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:(џџџџџџџџџџџџџџџџџџ:f b
>
_output_shapes,
*:(џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
б
Й
!__inference_internal_grad_fn_5401
result_grads_0
result_grads_1
result_grads_2
result_grads_36
2ones_like_shape_while_spiking_activation_cell_relu
identityq
ones_like/ShapeShape2ones_like_shape_while_spiking_activation_cell_relu*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?x
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЛ
4gradient_tape/while/spiking_activation_cell/ReluGradReluGradones_like:output:02ones_like_shape_while_spiking_activation_cell_relu*
T0*(
_output_shapes
:џџџџџџџџџ
mulMul@gradient_tape/while/spiking_activation_cell/ReluGrad:backprops:0result_grads_0*
T0*(
_output_shapes
:џџџџџџџџџP
IdentityIdentitymul:z:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*w
_input_shapesf
d:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:X T
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_0:XT
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_1:XT
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_2:XT
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_3:.*
(
_output_shapes
:џџџџџџџџџ
U
б
 __inference__traced_restore_5626
file_prefix2
assignvariableop_dense_1_kernel:	
-
assignvariableop_1_dense_1_bias:
&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: ?
*assignvariableop_7_time_distributed_kernel:7
(assignvariableop_8_time_distributed_bias:	"
assignvariableop_9_total: #
assignvariableop_10_count: %
assignvariableop_11_total_1: %
assignvariableop_12_count_1: <
)assignvariableop_13_adam_dense_1_kernel_m:	
5
'assignvariableop_14_adam_dense_1_bias_m:
G
2assignvariableop_15_adam_time_distributed_kernel_m:?
0assignvariableop_16_adam_time_distributed_bias_m:	<
)assignvariableop_17_adam_dense_1_kernel_v:	
5
'assignvariableop_18_adam_dense_1_bias_v:
G
2assignvariableop_19_adam_time_distributed_kernel_v:?
0assignvariableop_20_adam_time_distributed_bias_v:	
identity_22ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_3ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9ф

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*

value
B§	B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value6B4B B B B B B B B B B B B B B B B B B B B B B 
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*l
_output_shapesZ
X::::::::::::::::::::::*$
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOpassignvariableop_dense_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp*assignvariableop_7_time_distributed_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp(assignvariableop_8_time_distributed_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOpassignvariableop_9_totalIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOpassignvariableop_10_countIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_total_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOpassignvariableop_12_count_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp)assignvariableop_13_adam_dense_1_kernel_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOp'assignvariableop_14_adam_dense_1_bias_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:Ѓ
AssignVariableOp_15AssignVariableOp2assignvariableop_15_adam_time_distributed_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:Ё
AssignVariableOp_16AssignVariableOp0assignvariableop_16_adam_time_distributed_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp)assignvariableop_17_adam_dense_1_kernel_vIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp'assignvariableop_18_adam_dense_1_bias_vIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Ѓ
AssignVariableOp_19AssignVariableOp2assignvariableop_19_adam_time_distributed_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:Ё
AssignVariableOp_20AssignVariableOp0assignvariableop_20_adam_time_distributed_bias_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 
Identity_21Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_22IdentityIdentity_21:output:0^NoOp_1*
T0*
_output_shapes
: 
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_22Identity_22:output:0*?
_input_shapes.
,: : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
К
j
1__inference_spiking_activation_layer_call_fn_4734

inputs
identityЂStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_spiking_activation_layer_call_and_return_conditional_losses_4194}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџџџџџџџџџџ22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Њ

while_cond_5195
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_12
.while_while_cond_5195___redundant_placeholder0
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
": : : : :џџџџџџџџџ: :: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:

 
/__inference_time_distributed_layer_call_fn_4682

inputs
unknown:
	unknown_0:	
identityЂStatefulPartitionedCall№
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_time_distributed_layer_call_and_return_conditional_losses_3612}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: џџџџџџџџџџџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
6
_output_shapes$
": џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Њ{

D__inference_sequential_layer_call_and_return_conditional_losses_4631

inputsJ
5time_distributed_dense_matmul_readvariableop_resource:E
6time_distributed_dense_biasadd_readvariableop_resource:	9
&dense_1_matmul_readvariableop_resource:	
5
'dense_1_biasadd_readvariableop_resource:

identityЂdense_1/BiasAdd/ReadVariableOpЂdense_1/MatMul/ReadVariableOpЂ-time_distributed/dense/BiasAdd/ReadVariableOpЂ,time_distributed/dense/MatMul/ReadVariableOpC
reshape/ShapeShapeinputs*
T0*
_output_shapes
:e
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:љ
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ[
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
valueB	 :Џ
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:
reshape/ReshapeReshapeinputsreshape/Reshape/shape:output:0*
T0*6
_output_shapes$
": џџџџџџџџџџџџџџџџџџ^
time_distributed/ShapeShapereshape/Reshape:output:0*
T0*
_output_shapes
:n
$time_distributed/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:p
&time_distributed/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&time_distributed/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:І
time_distributed/strided_sliceStridedSlicetime_distributed/Shape:output:0-time_distributed/strided_slice/stack:output:0/time_distributed/strided_slice/stack_1:output:0/time_distributed/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ @  
time_distributed/ReshapeReshapereshape/Reshape:output:0'time_distributed/Reshape/shape:output:0*
T0*)
_output_shapes
:џџџџџџџџџЅ
,time_distributed/dense/MatMul/ReadVariableOpReadVariableOp5time_distributed_dense_matmul_readvariableop_resource*!
_output_shapes
:*
dtype0Г
time_distributed/dense/MatMulMatMul!time_distributed/Reshape:output:04time_distributed/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЁ
-time_distributed/dense/BiasAdd/ReadVariableOpReadVariableOp6time_distributed_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0М
time_distributed/dense/BiasAddBiasAdd'time_distributed/dense/MatMul:product:05time_distributed/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџm
"time_distributed/Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџe
"time_distributed/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :й
 time_distributed/Reshape_1/shapePack+time_distributed/Reshape_1/shape/0:output:0'time_distributed/strided_slice:output:0+time_distributed/Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:Й
time_distributed/Reshape_1Reshape'time_distributed/dense/BiasAdd:output:0)time_distributed/Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџq
 time_distributed/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ @  
time_distributed/Reshape_2Reshapereshape/Reshape:output:0)time_distributed/Reshape_2/shape:output:0*
T0*)
_output_shapes
:џџџџџџџџџk
spiking_activation/ShapeShape#time_distributed/Reshape_1:output:0*
T0*
_output_shapes
:p
&spiking_activation/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(spiking_activation/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(spiking_activation/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:А
 spiking_activation/strided_sliceStridedSlice!spiking_activation/Shape:output:0/spiking_activation/strided_slice/stack:output:01spiking_activation/strided_slice/stack_1:output:01spiking_activation/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
'spiking_activation/random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB g
%spiking_activation/random_uniform/minConst*
_output_shapes
: *
dtype0*
value	B : k
%spiking_activation/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB :џџџџє
!spiking_activation/random_uniformRandomUniformInt0spiking_activation/random_uniform/shape:output:0.spiking_activation/random_uniform/min:output:0.spiking_activation/random_uniform/max:output:0*
T0*

Tout0*
_output_shapes
: v
3spiking_activation/stateless_random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value
B :а
1spiking_activation/stateless_random_uniform/shapePack)spiking_activation/strided_slice:output:0<spiking_activation/stateless_random_uniform/shape/1:output:0*
N*
T0*
_output_shapes
:t
/spiking_activation/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    t
/spiking_activation/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?л
Mspiking_activation/stateless_random_uniform/StatelessRandomGetKeyCounter/seedPack*spiking_activation/random_uniform:output:0*spiking_activation/random_uniform:output:0*
N*
T0*
_output_shapes
:я
Hspiking_activation/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterVspiking_activation/stateless_random_uniform/StatelessRandomGetKeyCounter/seed:output:0*
Tseed0* 
_output_shapes
::
Hspiking_activation/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :Н
Dspiking_activation/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2:spiking_activation/stateless_random_uniform/shape:output:0Nspiking_activation/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0Rspiking_activation/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Qspiking_activation/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*(
_output_shapes
:џџџџџџџџџЫ
/spiking_activation/stateless_random_uniform/subSub8spiking_activation/stateless_random_uniform/max:output:08spiking_activation/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: э
/spiking_activation/stateless_random_uniform/mulMulMspiking_activation/stateless_random_uniform/StatelessRandomUniformV2:output:03spiking_activation/stateless_random_uniform/sub:z:0*
T0*(
_output_shapes
:џџџџџџџџџж
+spiking_activation/stateless_random_uniformAddV23spiking_activation/stateless_random_uniform/mul:z:08spiking_activation/stateless_random_uniform/min:output:0*
T0*(
_output_shapes
:џџџџџџџџџv
!spiking_activation/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          К
spiking_activation/transpose	Transpose#time_distributed/Reshape_1:output:0*spiking_activation/transpose/perm:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџj
spiking_activation/Shape_1Shape spiking_activation/transpose:y:0*
T0*
_output_shapes
:r
(spiking_activation/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*spiking_activation/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*spiking_activation/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:К
"spiking_activation/strided_slice_1StridedSlice#spiking_activation/Shape_1:output:01spiking_activation/strided_slice_1/stack:output:03spiking_activation/strided_slice_1/stack_1:output:03spiking_activation/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masky
.spiking_activation/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџэ
 spiking_activation/TensorArrayV2TensorListReserve7spiking_activation/TensorArrayV2/element_shape:output:0+spiking_activation/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
Hspiking_activation/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
:spiking_activation/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor spiking_activation/transpose:y:0Qspiking_activation/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвr
(spiking_activation/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*spiking_activation/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*spiking_activation/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Щ
"spiking_activation/strided_slice_2StridedSlice spiking_activation/transpose:y:01spiking_activation/strided_slice_2/stack:output:03spiking_activation/strided_slice_2/stack_1:output:03spiking_activation/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_masks
1spiking_activation/spiking_activation_cell/Cast/xConst*
_output_shapes
: *
dtype0
*
value	B
 Zy
7spiking_activation/spiking_activation_cell/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Zй
5spiking_activation/spiking_activation_cell/LogicalAnd
LogicalAnd:spiking_activation/spiking_activation_cell/Cast/x:output:0@spiking_activation/spiking_activation_cell/LogicalAnd/y:output:0*
_output_shapes
: 
/spiking_activation/spiking_activation_cell/ReluRelu+spiking_activation/strided_slice_2:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
0spiking_activation/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ё
"spiking_activation/TensorArrayV2_1TensorListReserve9spiking_activation/TensorArrayV2_1/element_shape:output:0+spiking_activation/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвY
spiking_activation/timeConst*
_output_shapes
: *
dtype0*
value	B : v
+spiking_activation/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџg
%spiking_activation/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ў
spiking_activation/whileStatelessWhile.spiking_activation/while/loop_counter:output:04spiking_activation/while/maximum_iterations:output:0 spiking_activation/time:output:0+spiking_activation/TensorArrayV2_1:handle:0/spiking_activation/stateless_random_uniform:z:0+spiking_activation/strided_slice_1:output:0Jspiking_activation/TensorArrayUnstack/TensorListFromTensor:output_handle:0*
T
	2*
_lower_using_switch_merge(*
_num_original_outputs*4
_output_shapes"
 : : : : :џџџџџџџџџ: : * 
_read_only_resource_inputs
 *
_stateful_parallelism( *.
body&R$
"spiking_activation_while_body_4570*.
cond&R$
"spiking_activation_while_cond_4569*3
output_shapes"
 : : : : :џџџџџџџџџ: : *
parallel_iterations 
Cspiking_activation/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
5spiking_activation/TensorArrayV2Stack/TensorListStackTensorListStack!spiking_activation/while:output:3Lspiking_activation/TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ*
element_dtype0{
(spiking_activation/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџt
*spiking_activation/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: t
*spiking_activation/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ч
"spiking_activation/strided_slice_3StridedSlice>spiking_activation/TensorArrayV2Stack/TensorListStack:tensor:01spiking_activation/strided_slice_3/stack:output:03spiking_activation/strided_slice_3/stack_1:output:03spiking_activation/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskx
#spiking_activation/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          й
spiking_activation/transpose_1	Transpose>spiking_activation/TensorArrayV2Stack/TensorListStack:tensor:0,spiking_activation/transpose_1/perm:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџq
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :Ж
global_average_pooling1d/MeanMean"spiking_activation/transpose_1:y:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype0
dense_1/MatMulMatMul&global_average_pooling1d/Mean:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ

dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
g
IdentityIdentitydense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
ц
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp.^time_distributed/dense/BiasAdd/ReadVariableOp-^time_distributed/dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:(џџџџџџџџџџџџџџџџџџ: : : : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2^
-time_distributed/dense/BiasAdd/ReadVariableOp-time_distributed/dense/BiasAdd/ReadVariableOp2\
,time_distributed/dense/MatMul/ReadVariableOp,time_distributed/dense/MatMul/ReadVariableOp:f b
>
_output_shapes,
*:(џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ш	
ѓ
A__inference_dense_1_layer_call_and_return_conditional_losses_4058

inputs1
matmul_readvariableop_resource:	
-
biasadd_readvariableop_resource:

identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ђJ
^
=__inference_rnn_layer_call_and_return_conditional_losses_5149
inputs_0
identity=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB T
random_uniform/minConst*
_output_shapes
: *
dtype0*
value	B : X
random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB :џџџџЈ
random_uniformRandomUniformIntrandom_uniform/shape:output:0random_uniform/min:output:0random_uniform/max:output:0*
T0*

Tout0*
_output_shapes
: c
 stateless_random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value
B :
stateless_random_uniform/shapePackstrided_slice:output:0)stateless_random_uniform/shape/1:output:0*
N*
T0*
_output_shapes
:a
stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    a
stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ђ
:stateless_random_uniform/StatelessRandomGetKeyCounter/seedPackrandom_uniform:output:0random_uniform:output:0*
N*
T0*
_output_shapes
:Щ
5stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterCstateless_random_uniform/StatelessRandomGetKeyCounter/seed:output:0*
Tseed0* 
_output_shapes
::w
5stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :о
1stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2'stateless_random_uniform/shape:output:0;stateless_random_uniform/StatelessRandomGetKeyCounter:key:0?stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0>stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*(
_output_shapes
:џџџџџџџџџ
stateless_random_uniform/subSub%stateless_random_uniform/max:output:0%stateless_random_uniform/min:output:0*
T0*
_output_shapes
: Д
stateless_random_uniform/mulMul:stateless_random_uniform/StatelessRandomUniformV2:output:0 stateless_random_uniform/sub:z:0*
T0*(
_output_shapes
:џџџџџџџџџ
stateless_random_uniformAddV2 stateless_random_uniform/mul:z:0%stateless_random_uniform/min:output:0*
T0*(
_output_shapes
:џџџџџџџџџc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          y
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ъ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask`
spiking_activation_cell/Cast/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z f
$spiking_activation_cell/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z 
"spiking_activation_cell/LogicalAnd
LogicalAnd'spiking_activation_cell/Cast/x:output:0-spiking_activation_cell/LogicalAnd/y:output:0*
_output_shapes
: q
spiking_activation_cell/ReluRelustrided_slice_2:output:0*
T0*(
_output_shapes
:џџџџџџџџџb
spiking_activation_cell/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:Љ
spiking_activation_cell/mulMul*spiking_activation_cell/Relu:activations:0&spiking_activation_cell/mul/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
spiking_activation_cell/addAddV2stateless_random_uniform:z:0spiking_activation_cell/mul:z:0*
T0*(
_output_shapes
:џџџџџџџџџz
spiking_activation_cell/FloorFloorspiking_activation_cell/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџ
spiking_activation_cell/subSubspiking_activation_cell/add:z:0!spiking_activation_cell/Floor:y:0*
T0*(
_output_shapes
:џџџџџџџџџf
!spiking_activation_cell/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:Ќ
spiking_activation_cell/truedivRealDiv!spiking_activation_cell/Floor:y:0*spiking_activation_cell/truediv/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
 spiking_activation_cell/IdentityIdentity#spiking_activation_cell/truediv:z:0*
T0*(
_output_shapes
:џџџџџџџџџ
"spiking_activation_cell/Identity_1Identityspiking_activation_cell/sub:z:0*
T0*(
_output_shapes
:џџџџџџџџџЭ
!spiking_activation_cell/IdentityN	IdentityN#spiking_activation_cell/truediv:z:0spiking_activation_cell/sub:z:0strided_slice_2:output:0stateless_random_uniform:z:0*
T
2**
_gradient_op_typeCustomGradient-5063*d
_output_shapesR
P:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Р
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0stateless_random_uniform:z:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*
T
	2*
_lower_using_switch_merge(*
_num_original_outputs*4
_output_shapes"
 : : : : :џџџџџџџџџ: : * 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_5082*
condR
while_cond_5081*3
output_shapes"
 : : : : :џџџџџџџџџ: : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ь
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџe
IdentityIdentitytranspose_1:y:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџџџџџџџџџџ:_ [
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs/0
Т
№
while_body_3693
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ї
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџ*
element_dtype0Г
-while/spiking_activation_cell/PartitionedCallPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_spiking_activation_cell_layer_call_and_return_conditional_losses_3686п
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder6while/spiking_activation_cell/PartitionedCall:output:0*
_output_shapes
: *
element_dtype0:щшвM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: L
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: :щшв
while/Identity_4Identity6while/spiking_activation_cell/PartitionedCall:output:1*
T0*(
_output_shapes
:џџџџџџџџџ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 : : : : :џџџџџџџџџ: : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
К

D__inference_sequential_layer_call_and_return_conditional_losses_4495

inputsJ
5time_distributed_dense_matmul_readvariableop_resource:E
6time_distributed_dense_biasadd_readvariableop_resource:	9
&dense_1_matmul_readvariableop_resource:	
5
'dense_1_biasadd_readvariableop_resource:

identityЂdense_1/BiasAdd/ReadVariableOpЂdense_1/MatMul/ReadVariableOpЂ-time_distributed/dense/BiasAdd/ReadVariableOpЂ,time_distributed/dense/MatMul/ReadVariableOpC
reshape/ShapeShapeinputs*
T0*
_output_shapes
:e
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:љ
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ[
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
valueB	 :Џ
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:
reshape/ReshapeReshapeinputsreshape/Reshape/shape:output:0*
T0*6
_output_shapes$
": џџџџџџџџџџџџџџџџџџ^
time_distributed/ShapeShapereshape/Reshape:output:0*
T0*
_output_shapes
:n
$time_distributed/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:p
&time_distributed/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&time_distributed/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:І
time_distributed/strided_sliceStridedSlicetime_distributed/Shape:output:0-time_distributed/strided_slice/stack:output:0/time_distributed/strided_slice/stack_1:output:0/time_distributed/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ @  
time_distributed/ReshapeReshapereshape/Reshape:output:0'time_distributed/Reshape/shape:output:0*
T0*)
_output_shapes
:џџџџџџџџџЅ
,time_distributed/dense/MatMul/ReadVariableOpReadVariableOp5time_distributed_dense_matmul_readvariableop_resource*!
_output_shapes
:*
dtype0Г
time_distributed/dense/MatMulMatMul!time_distributed/Reshape:output:04time_distributed/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЁ
-time_distributed/dense/BiasAdd/ReadVariableOpReadVariableOp6time_distributed_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0М
time_distributed/dense/BiasAddBiasAdd'time_distributed/dense/MatMul:product:05time_distributed/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџm
"time_distributed/Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџe
"time_distributed/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :й
 time_distributed/Reshape_1/shapePack+time_distributed/Reshape_1/shape/0:output:0'time_distributed/strided_slice:output:0+time_distributed/Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:Й
time_distributed/Reshape_1Reshape'time_distributed/dense/BiasAdd:output:0)time_distributed/Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџq
 time_distributed/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ @  
time_distributed/Reshape_2Reshapereshape/Reshape:output:0)time_distributed/Reshape_2/shape:output:0*
T0*)
_output_shapes
:џџџџџџџџџk
spiking_activation/ShapeShape#time_distributed/Reshape_1:output:0*
T0*
_output_shapes
:p
&spiking_activation/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(spiking_activation/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(spiking_activation/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:А
 spiking_activation/strided_sliceStridedSlice!spiking_activation/Shape:output:0/spiking_activation/strided_slice/stack:output:01spiking_activation/strided_slice/stack_1:output:01spiking_activation/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
'spiking_activation/random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB g
%spiking_activation/random_uniform/minConst*
_output_shapes
: *
dtype0*
value	B : k
%spiking_activation/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB :џџџџє
!spiking_activation/random_uniformRandomUniformInt0spiking_activation/random_uniform/shape:output:0.spiking_activation/random_uniform/min:output:0.spiking_activation/random_uniform/max:output:0*
T0*

Tout0*
_output_shapes
: v
3spiking_activation/stateless_random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value
B :а
1spiking_activation/stateless_random_uniform/shapePack)spiking_activation/strided_slice:output:0<spiking_activation/stateless_random_uniform/shape/1:output:0*
N*
T0*
_output_shapes
:t
/spiking_activation/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    t
/spiking_activation/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?л
Mspiking_activation/stateless_random_uniform/StatelessRandomGetKeyCounter/seedPack*spiking_activation/random_uniform:output:0*spiking_activation/random_uniform:output:0*
N*
T0*
_output_shapes
:я
Hspiking_activation/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterVspiking_activation/stateless_random_uniform/StatelessRandomGetKeyCounter/seed:output:0*
Tseed0* 
_output_shapes
::
Hspiking_activation/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :Н
Dspiking_activation/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2:spiking_activation/stateless_random_uniform/shape:output:0Nspiking_activation/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0Rspiking_activation/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Qspiking_activation/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*(
_output_shapes
:џџџџџџџџџЫ
/spiking_activation/stateless_random_uniform/subSub8spiking_activation/stateless_random_uniform/max:output:08spiking_activation/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: э
/spiking_activation/stateless_random_uniform/mulMulMspiking_activation/stateless_random_uniform/StatelessRandomUniformV2:output:03spiking_activation/stateless_random_uniform/sub:z:0*
T0*(
_output_shapes
:џџџџџџџџџж
+spiking_activation/stateless_random_uniformAddV23spiking_activation/stateless_random_uniform/mul:z:08spiking_activation/stateless_random_uniform/min:output:0*
T0*(
_output_shapes
:џџџџџџџџџv
!spiking_activation/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          К
spiking_activation/transpose	Transpose#time_distributed/Reshape_1:output:0*spiking_activation/transpose/perm:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџj
spiking_activation/Shape_1Shape spiking_activation/transpose:y:0*
T0*
_output_shapes
:r
(spiking_activation/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*spiking_activation/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*spiking_activation/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:К
"spiking_activation/strided_slice_1StridedSlice#spiking_activation/Shape_1:output:01spiking_activation/strided_slice_1/stack:output:03spiking_activation/strided_slice_1/stack_1:output:03spiking_activation/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masky
.spiking_activation/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџэ
 spiking_activation/TensorArrayV2TensorListReserve7spiking_activation/TensorArrayV2/element_shape:output:0+spiking_activation/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
Hspiking_activation/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
:spiking_activation/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor spiking_activation/transpose:y:0Qspiking_activation/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвr
(spiking_activation/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*spiking_activation/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*spiking_activation/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Щ
"spiking_activation/strided_slice_2StridedSlice spiking_activation/transpose:y:01spiking_activation/strided_slice_2/stack:output:03spiking_activation/strided_slice_2/stack_1:output:03spiking_activation/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_masks
1spiking_activation/spiking_activation_cell/Cast/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z y
7spiking_activation/spiking_activation_cell/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Zй
5spiking_activation/spiking_activation_cell/LogicalAnd
LogicalAnd:spiking_activation/spiking_activation_cell/Cast/x:output:0@spiking_activation/spiking_activation_cell/LogicalAnd/y:output:0*
_output_shapes
: 
/spiking_activation/spiking_activation_cell/ReluRelu+spiking_activation/strided_slice_2:output:0*
T0*(
_output_shapes
:џџџџџџџџџu
0spiking_activation/spiking_activation_cell/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:т
.spiking_activation/spiking_activation_cell/mulMul=spiking_activation/spiking_activation_cell/Relu:activations:09spiking_activation/spiking_activation_cell/mul/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЯ
.spiking_activation/spiking_activation_cell/addAddV2/spiking_activation/stateless_random_uniform:z:02spiking_activation/spiking_activation_cell/mul:z:0*
T0*(
_output_shapes
:џџџџџџџџџ 
0spiking_activation/spiking_activation_cell/FloorFloor2spiking_activation/spiking_activation_cell/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџв
.spiking_activation/spiking_activation_cell/subSub2spiking_activation/spiking_activation_cell/add:z:04spiking_activation/spiking_activation_cell/Floor:y:0*
T0*(
_output_shapes
:џџџџџџџџџy
4spiking_activation/spiking_activation_cell/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:х
2spiking_activation/spiking_activation_cell/truedivRealDiv4spiking_activation/spiking_activation_cell/Floor:y:0=spiking_activation/spiking_activation_cell/truediv/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЊ
3spiking_activation/spiking_activation_cell/IdentityIdentity6spiking_activation/spiking_activation_cell/truediv:z:0*
T0*(
_output_shapes
:џџџџџџџџџЈ
5spiking_activation/spiking_activation_cell/Identity_1Identity2spiking_activation/spiking_activation_cell/sub:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
4spiking_activation/spiking_activation_cell/IdentityN	IdentityN6spiking_activation/spiking_activation_cell/truediv:z:02spiking_activation/spiking_activation_cell/sub:z:0+spiking_activation/strided_slice_2:output:0/spiking_activation/stateless_random_uniform:z:0*
T
2**
_gradient_op_typeCustomGradient-4401*d
_output_shapesR
P:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ
0spiking_activation/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   ё
"spiking_activation/TensorArrayV2_1TensorListReserve9spiking_activation/TensorArrayV2_1/element_shape:output:0+spiking_activation/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвY
spiking_activation/timeConst*
_output_shapes
: *
dtype0*
value	B : v
+spiking_activation/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџg
%spiking_activation/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ў
spiking_activation/whileStatelessWhile.spiking_activation/while/loop_counter:output:04spiking_activation/while/maximum_iterations:output:0 spiking_activation/time:output:0+spiking_activation/TensorArrayV2_1:handle:0/spiking_activation/stateless_random_uniform:z:0+spiking_activation/strided_slice_1:output:0Jspiking_activation/TensorArrayUnstack/TensorListFromTensor:output_handle:0*
T
	2*
_lower_using_switch_merge(*
_num_original_outputs*4
_output_shapes"
 : : : : :џџџџџџџџџ: : * 
_read_only_resource_inputs
 *
_stateful_parallelism( *.
body&R$
"spiking_activation_while_body_4420*.
cond&R$
"spiking_activation_while_cond_4419*3
output_shapes"
 : : : : :џџџџџџџџџ: : *
parallel_iterations 
Cspiking_activation/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
5spiking_activation/TensorArrayV2Stack/TensorListStackTensorListStack!spiking_activation/while:output:3Lspiking_activation/TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ*
element_dtype0{
(spiking_activation/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџt
*spiking_activation/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: t
*spiking_activation/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ч
"spiking_activation/strided_slice_3StridedSlice>spiking_activation/TensorArrayV2Stack/TensorListStack:tensor:01spiking_activation/strided_slice_3/stack:output:03spiking_activation/strided_slice_3/stack_1:output:03spiking_activation/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_maskx
#spiking_activation/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          й
spiking_activation/transpose_1	Transpose>spiking_activation/TensorArrayV2Stack/TensorListStack:tensor:0,spiking_activation/transpose_1/perm:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџq
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :Ж
global_average_pooling1d/MeanMean"spiking_activation/transpose_1:y:08global_average_pooling1d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype0
dense_1/MatMulMatMul&global_average_pooling1d/Mean:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ

dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
g
IdentityIdentitydense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
ц
NoOpNoOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp.^time_distributed/dense/BiasAdd/ReadVariableOp-^time_distributed/dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:(џџџџџџџџџџџџџџџџџџ: : : : 2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2^
-time_distributed/dense/BiasAdd/ReadVariableOp-time_distributed/dense/BiasAdd/ReadVariableOp2\
,time_distributed/dense/MatMul/ReadVariableOp,time_distributed/dense/MatMul/ReadVariableOp:f b
>
_output_shapes,
*:(џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
§
ѓ
D__inference_sequential_layer_call_and_return_conditional_losses_4299
reshape_input*
time_distributed_4284:$
time_distributed_4286:	
dense_1_4293:	

dense_1_4295:

identityЂdense_1/StatefulPartitionedCallЂ*spiking_activation/StatefulPartitionedCallЂ(time_distributed/StatefulPartitionedCallЭ
reshape/PartitionedCallPartitionedCallreshape_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *6
_output_shapes$
": џџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_reshape_layer_call_and_return_conditional_losses_3908Е
(time_distributed/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0time_distributed_4284time_distributed_4286*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_time_distributed_layer_call_and_return_conditional_losses_3612o
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ @  Ђ
time_distributed/ReshapeReshape reshape/PartitionedCall:output:0'time_distributed/Reshape/shape:output:0*
T0*)
_output_shapes
:џџџџџџџџџ
*spiking_activation/StatefulPartitionedCallStatefulPartitionedCall1time_distributed/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_spiking_activation_layer_call_and_return_conditional_losses_4194
(global_average_pooling1d/PartitionedCallPartitionedCall3spiking_activation/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_3885
dense_1/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0dense_1_4293dense_1_4295*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_4058w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
Р
NoOpNoOp ^dense_1/StatefulPartitionedCall+^spiking_activation/StatefulPartitionedCall)^time_distributed/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:(џџџџџџџџџџџџџџџџџџ: : : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2X
*spiking_activation/StatefulPartitionedCall*spiking_activation/StatefulPartitionedCall2T
(time_distributed/StatefulPartitionedCall(time_distributed/StatefulPartitionedCall:m i
>
_output_shapes,
*:(џџџџџџџџџџџџџџџџџџ
'
_user_specified_namereshape_input
ь
p
6__inference_spiking_activation_cell_layer_call_fn_5265

inputs
states_0
identity

identity_1р
PartitionedCallPartitionedCallinputsstates_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_spiking_activation_cell_layer_call_and_return_conditional_losses_3769a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџc

Identity_1IdentityPartitionedCall:output:1*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџ:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:RN
(
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
states/0
Г
Г
!__inference_internal_grad_fn_5443
result_grads_0
result_grads_1
result_grads_2
result_grads_30
,ones_like_shape_spiking_activation_cell_relu
identityk
ones_like/ShapeShape,ones_like_shape_spiking_activation_cell_relu*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?x
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЏ
.gradient_tape/spiking_activation_cell/ReluGradReluGradones_like:output:0,ones_like_shape_spiking_activation_cell_relu*
T0*(
_output_shapes
:џџџџџџџџџ
mulMul:gradient_tape/spiking_activation_cell/ReluGrad:backprops:0result_grads_0*
T0*(
_output_shapes
:џџџџџџџџџP
IdentityIdentitymul:z:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*w
_input_shapesf
d:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:X T
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_0:XT
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_1:XT
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_2:XT
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_3:.*
(
_output_shapes
:џџџџџџџџџ
Њ

while_cond_3977
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_12
.while_while_cond_3977___redundant_placeholder0
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
": : : : :џџџџџџџџџ: :: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
Њ

while_cond_5081
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_12
.while_while_cond_5081___redundant_placeholder0
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
": : : : :џџџџџџџџџ: :: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
б
Й
!__inference_internal_grad_fn_5457
result_grads_0
result_grads_1
result_grads_2
result_grads_36
2ones_like_shape_while_spiking_activation_cell_relu
identityq
ones_like/ShapeShape2ones_like_shape_while_spiking_activation_cell_relu*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?x
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЛ
4gradient_tape/while/spiking_activation_cell/ReluGradReluGradones_like:output:02ones_like_shape_while_spiking_activation_cell_relu*
T0*(
_output_shapes
:џџџџџџџџџ
mulMul@gradient_tape/while/spiking_activation_cell/ReluGrad:backprops:0result_grads_0*
T0*(
_output_shapes
:џџџџџџџџџP
IdentityIdentitymul:z:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*w
_input_shapesf
d:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:X T
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_0:XT
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_1:XT
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_2:XT
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_3:.*
(
_output_shapes
:џџџџџџџџџ
Њ

while_cond_4908
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_12
.while_while_cond_4908___redundant_placeholder0
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
": : : : :џџџџџџџџџ: :: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
К	
ћ
"spiking_activation_while_cond_4569B
>spiking_activation_while_spiking_activation_while_loop_counterH
Dspiking_activation_while_spiking_activation_while_maximum_iterations(
$spiking_activation_while_placeholder*
&spiking_activation_while_placeholder_1*
&spiking_activation_while_placeholder_2D
@spiking_activation_while_less_spiking_activation_strided_slice_1X
Tspiking_activation_while_spiking_activation_while_cond_4569___redundant_placeholder0%
!spiking_activation_while_identity
Ў
spiking_activation/while/LessLess$spiking_activation_while_placeholder@spiking_activation_while_less_spiking_activation_strided_slice_1*
T0*
_output_shapes
: q
!spiking_activation/while/IdentityIdentity!spiking_activation/while/Less:z:0*
T0
*
_output_shapes
: "O
!spiking_activation_while_identity*spiking_activation/while/Identity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
": : : : :џџџџџџџџџ: :: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:

Х
J__inference_time_distributed_layer_call_and_return_conditional_losses_3573

inputs

dense_3563:

dense_3565:	
identityЂdense/StatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ @  f
ReshapeReshapeinputsReshape/shape:output:0*
T0*)
_output_shapes
:џџџџџџџџџь
dense/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0
dense_3563
dense_3565*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_3562\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshape&dense/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџo
IdentityIdentityReshape_1:output:0^NoOp*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџf
NoOpNoOp^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: џџџџџџџџџџџџџџџџџџ: : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:^ Z
6
_output_shapes$
": џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
б	
є
?__inference_dense_layer_call_and_return_conditional_losses_3562

inputs3
matmul_readvariableop_resource:.
biasadd_readvariableop_resource:	
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpw
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ`
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
§
ѓ
D__inference_sequential_layer_call_and_return_conditional_losses_4280
reshape_input*
time_distributed_4265:$
time_distributed_4267:	
dense_1_4274:	

dense_1_4276:

identityЂdense_1/StatefulPartitionedCallЂ*spiking_activation/StatefulPartitionedCallЂ(time_distributed/StatefulPartitionedCallЭ
reshape/PartitionedCallPartitionedCallreshape_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *6
_output_shapes$
": џџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_reshape_layer_call_and_return_conditional_losses_3908Е
(time_distributed/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0time_distributed_4265time_distributed_4267*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_time_distributed_layer_call_and_return_conditional_losses_3573o
time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ @  Ђ
time_distributed/ReshapeReshape reshape/PartitionedCall:output:0'time_distributed/Reshape/shape:output:0*
T0*)
_output_shapes
:џџџџџџџџџ
*spiking_activation/StatefulPartitionedCallStatefulPartitionedCall1time_distributed/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_spiking_activation_layer_call_and_return_conditional_losses_4045
(global_average_pooling1d/PartitionedCallPartitionedCall3spiking_activation/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_3885
dense_1/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling1d/PartitionedCall:output:0dense_1_4274dense_1_4276*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_4058w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
Р
NoOpNoOp ^dense_1/StatefulPartitionedCall+^spiking_activation/StatefulPartitionedCall)^time_distributed/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:(џџџџџџџџџџџџџџџџџџ: : : : 2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2X
*spiking_activation/StatefulPartitionedCall*spiking_activation/StatefulPartitionedCall2T
(time_distributed/StatefulPartitionedCall(time_distributed/StatefulPartitionedCall:m i
>
_output_shapes,
*:(џџџџџџџџџџџџџџџџџџ
'
_user_specified_namereshape_input
с
и
)__inference_sequential_layer_call_fn_4076
reshape_input
unknown:
	unknown_0:	
	unknown_1:	

	unknown_2:

identityЂStatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallreshape_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_4065o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:(џџџџџџџџџџџџџџџџџџ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:m i
>
_output_shapes,
*:(џџџџџџџџџџџџџџџџџџ
'
_user_specified_namereshape_input
Ц

Q__inference_spiking_activation_cell_layer_call_and_return_conditional_losses_3769

inputs

states
identity

identity_1H
Cast/xConst*
_output_shapes
: *
dtype0
*
value	B
 ZN
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 ZX

LogicalAnd
LogicalAndCast/x:output:0LogicalAnd/y:output:0*
_output_shapes
: G
ReluReluinputs*
T0*(
_output_shapes
:џџџџџџџџџ[
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџQ

Identity_1Identitystates*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџ:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:PL
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_namestates
џ
№
while_body_4141
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ї
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџ*
element_dtype0f
$while/spiking_activation_cell/Cast/xConst*
_output_shapes
: *
dtype0
*
value	B
 Zl
*while/spiking_activation_cell/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 ZВ
(while/spiking_activation_cell/LogicalAnd
LogicalAnd-while/spiking_activation_cell/Cast/x:output:03while/spiking_activation_cell/LogicalAnd/y:output:0*
_output_shapes
: 
"while/spiking_activation_cell/ReluRelu0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*(
_output_shapes
:џџџџџџџџџй
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder0while/spiking_activation_cell/Relu:activations:0*
_output_shapes
: *
element_dtype0:щшвM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: L
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: :щшвd
while/Identity_4Identitywhile_placeholder_2*
T0*(
_output_shapes
:џџџџџџџџџ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 : : : : :џџџџџџџџџ: : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
њ
S
7__inference_global_average_pooling1d_layer_call_fn_4967

inputs
identityЩ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *[
fVRT
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_3885i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ:e a
=
_output_shapes+
):'џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
К	
ћ
"spiking_activation_while_cond_4419B
>spiking_activation_while_spiking_activation_while_loop_counterH
Dspiking_activation_while_spiking_activation_while_maximum_iterations(
$spiking_activation_while_placeholder*
&spiking_activation_while_placeholder_1*
&spiking_activation_while_placeholder_2D
@spiking_activation_while_less_spiking_activation_strided_slice_1X
Tspiking_activation_while_spiking_activation_while_cond_4419___redundant_placeholder0%
!spiking_activation_while_identity
Ў
spiking_activation/while/LessLess$spiking_activation_while_placeholder@spiking_activation_while_less_spiking_activation_strided_slice_1*
T0*
_output_shapes
: q
!spiking_activation/while/IdentityIdentity!spiking_activation/while/Less:z:0*
T0
*
_output_shapes
: "O
!spiking_activation_while_identity*spiking_activation/while/Identity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
": : : : :џџџџџџџџџ: :: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
У?
ъ
-sequential_spiking_activation_while_body_3463X
Tsequential_spiking_activation_while_sequential_spiking_activation_while_loop_counter^
Zsequential_spiking_activation_while_sequential_spiking_activation_while_maximum_iterations3
/sequential_spiking_activation_while_placeholder5
1sequential_spiking_activation_while_placeholder_15
1sequential_spiking_activation_while_placeholder_2W
Ssequential_spiking_activation_while_sequential_spiking_activation_strided_slice_1_0
sequential_spiking_activation_while_tensorarrayv2read_tensorlistgetitem_sequential_spiking_activation_tensorarrayunstack_tensorlistfromtensor_00
,sequential_spiking_activation_while_identity2
.sequential_spiking_activation_while_identity_12
.sequential_spiking_activation_while_identity_22
.sequential_spiking_activation_while_identity_32
.sequential_spiking_activation_while_identity_4U
Qsequential_spiking_activation_while_sequential_spiking_activation_strided_slice_1
sequential_spiking_activation_while_tensorarrayv2read_tensorlistgetitem_sequential_spiking_activation_tensorarrayunstack_tensorlistfromtensorІ
Usequential/spiking_activation/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   О
Gsequential/spiking_activation/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_spiking_activation_while_tensorarrayv2read_tensorlistgetitem_sequential_spiking_activation_tensorarrayunstack_tensorlistfromtensor_0/sequential_spiking_activation_while_placeholder^sequential/spiking_activation/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџ*
element_dtype0
Bsequential/spiking_activation/while/spiking_activation_cell/Cast/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 
Hsequential/spiking_activation/while/spiking_activation_cell/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z
Fsequential/spiking_activation/while/spiking_activation_cell/LogicalAnd
LogicalAndKsequential/spiking_activation/while/spiking_activation_cell/Cast/x:output:0Qsequential/spiking_activation/while/spiking_activation_cell/LogicalAnd/y:output:0*
_output_shapes
: Ы
@sequential/spiking_activation/while/spiking_activation_cell/ReluReluNsequential/spiking_activation/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*(
_output_shapes
:џџџџџџџџџ
Asequential/spiking_activation/while/spiking_activation_cell/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:
?sequential/spiking_activation/while/spiking_activation_cell/mulMulNsequential/spiking_activation/while/spiking_activation_cell/Relu:activations:0Jsequential/spiking_activation/while/spiking_activation_cell/mul/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџѓ
?sequential/spiking_activation/while/spiking_activation_cell/addAddV21sequential_spiking_activation_while_placeholder_2Csequential/spiking_activation/while/spiking_activation_cell/mul:z:0*
T0*(
_output_shapes
:џџџџџџџџџТ
Asequential/spiking_activation/while/spiking_activation_cell/FloorFloorCsequential/spiking_activation/while/spiking_activation_cell/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџ
?sequential/spiking_activation/while/spiking_activation_cell/subSubCsequential/spiking_activation/while/spiking_activation_cell/add:z:0Esequential/spiking_activation/while/spiking_activation_cell/Floor:y:0*
T0*(
_output_shapes
:џџџџџџџџџ
Esequential/spiking_activation/while/spiking_activation_cell/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:
Csequential/spiking_activation/while/spiking_activation_cell/truedivRealDivEsequential/spiking_activation/while/spiking_activation_cell/Floor:y:0Nsequential/spiking_activation/while/spiking_activation_cell/truediv/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЬ
Dsequential/spiking_activation/while/spiking_activation_cell/IdentityIdentityGsequential/spiking_activation/while/spiking_activation_cell/truediv:z:0*
T0*(
_output_shapes
:џџџџџџџџџЪ
Fsequential/spiking_activation/while/spiking_activation_cell/Identity_1IdentityCsequential/spiking_activation/while/spiking_activation_cell/sub:z:0*
T0*(
_output_shapes
:џџџџџџџџџ
Esequential/spiking_activation/while/spiking_activation_cell/IdentityN	IdentityNGsequential/spiking_activation/while/spiking_activation_cell/truediv:z:0Csequential/spiking_activation/while/spiking_activation_cell/sub:z:0Nsequential/spiking_activation/while/TensorArrayV2Read/TensorListGetItem:item:01sequential_spiking_activation_while_placeholder_2*
T
2**
_gradient_op_typeCustomGradient-3488*d
_output_shapesR
P:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџб
Hsequential/spiking_activation/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem1sequential_spiking_activation_while_placeholder_1/sequential_spiking_activation_while_placeholderNsequential/spiking_activation/while/spiking_activation_cell/IdentityN:output:0*
_output_shapes
: *
element_dtype0:щшвk
)sequential/spiking_activation/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Ж
'sequential/spiking_activation/while/addAddV2/sequential_spiking_activation_while_placeholder2sequential/spiking_activation/while/add/y:output:0*
T0*
_output_shapes
: m
+sequential/spiking_activation/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :п
)sequential/spiking_activation/while/add_1AddV2Tsequential_spiking_activation_while_sequential_spiking_activation_while_loop_counter4sequential/spiking_activation/while/add_1/y:output:0*
T0*
_output_shapes
: 
,sequential/spiking_activation/while/IdentityIdentity-sequential/spiking_activation/while/add_1:z:0*
T0*
_output_shapes
: З
.sequential/spiking_activation/while/Identity_1IdentityZsequential_spiking_activation_while_sequential_spiking_activation_while_maximum_iterations*
T0*
_output_shapes
: 
.sequential/spiking_activation/while/Identity_2Identity+sequential/spiking_activation/while/add:z:0*
T0*
_output_shapes
: Ш
.sequential/spiking_activation/while/Identity_3IdentityXsequential/spiking_activation/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: :щшвН
.sequential/spiking_activation/while/Identity_4IdentityNsequential/spiking_activation/while/spiking_activation_cell/IdentityN:output:1*
T0*(
_output_shapes
:џџџџџџџџџ"e
,sequential_spiking_activation_while_identity5sequential/spiking_activation/while/Identity:output:0"i
.sequential_spiking_activation_while_identity_17sequential/spiking_activation/while/Identity_1:output:0"i
.sequential_spiking_activation_while_identity_27sequential/spiking_activation/while/Identity_2:output:0"i
.sequential_spiking_activation_while_identity_37sequential/spiking_activation/while/Identity_3:output:0"i
.sequential_spiking_activation_while_identity_47sequential/spiking_activation/while/Identity_4:output:0"Ј
Qsequential_spiking_activation_while_sequential_spiking_activation_strided_slice_1Ssequential_spiking_activation_while_sequential_spiking_activation_strided_slice_1_0"Ђ
sequential_spiking_activation_while_tensorarrayv2read_tensorlistgetitem_sequential_spiking_activation_tensorarrayunstack_tensorlistfromtensorsequential_spiking_activation_while_tensorarrayv2read_tensorlistgetitem_sequential_spiking_activation_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 : : : : :џџџџџџџџџ: : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
џ
№
while_body_4909
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ї
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџ*
element_dtype0f
$while/spiking_activation_cell/Cast/xConst*
_output_shapes
: *
dtype0
*
value	B
 Zl
*while/spiking_activation_cell/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 ZВ
(while/spiking_activation_cell/LogicalAnd
LogicalAnd-while/spiking_activation_cell/Cast/x:output:03while/spiking_activation_cell/LogicalAnd/y:output:0*
_output_shapes
: 
"while/spiking_activation_cell/ReluRelu0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*(
_output_shapes
:џџџџџџџџџй
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder0while/spiking_activation_cell/Relu:activations:0*
_output_shapes
: *
element_dtype0:щшвM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: L
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: :щшвd
while/Identity_4Identitywhile_placeholder_2*
T0*(
_output_shapes
:џџџџџџџџџ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 : : : : :џџџџџџџџџ: : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
Ь
б
)__inference_sequential_layer_call_fn_4318

inputs
unknown:
	unknown_0:	
	unknown_1:	

	unknown_2:

identityЂStatefulPartitionedCallі
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_4065o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:(џџџџџџџџџџџџџџџџџџ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:f b
>
_output_shapes,
*:(џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Т

&__inference_dense_1_layer_call_fn_4982

inputs
unknown:	

	unknown_0:

identityЂStatefulPartitionedCallй
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_4058o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Њ

while_cond_4140
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_12
.while_while_cond_4140___redundant_placeholder0
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
": : : : :џџџџџџџџџ: :: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
Ђ

-sequential_spiking_activation_while_cond_3462X
Tsequential_spiking_activation_while_sequential_spiking_activation_while_loop_counter^
Zsequential_spiking_activation_while_sequential_spiking_activation_while_maximum_iterations3
/sequential_spiking_activation_while_placeholder5
1sequential_spiking_activation_while_placeholder_15
1sequential_spiking_activation_while_placeholder_2Z
Vsequential_spiking_activation_while_less_sequential_spiking_activation_strided_slice_1n
jsequential_spiking_activation_while_sequential_spiking_activation_while_cond_3462___redundant_placeholder00
,sequential_spiking_activation_while_identity
к
(sequential/spiking_activation/while/LessLess/sequential_spiking_activation_while_placeholderVsequential_spiking_activation_while_less_sequential_spiking_activation_strided_slice_1*
T0*
_output_shapes
: 
,sequential/spiking_activation/while/IdentityIdentity,sequential/spiking_activation/while/Less:z:0*
T0
*
_output_shapes
: "e
,sequential_spiking_activation_while_identity5sequential/spiking_activation/while/Identity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
": : : : :џџџџџџџџџ: :: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:
ц
B
&__inference_reshape_layer_call_fn_4651

inputs
identityО
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *6
_output_shapes$
": џџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_reshape_layer_call_and_return_conditional_losses_3908o
IdentityIdentityPartitionedCall:output:0*
T0*6
_output_shapes$
": џџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:(џџџџџџџџџџџџџџџџџџ:f b
>
_output_shapes,
*:(џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ш	
ѓ
A__inference_dense_1_layer_call_and_return_conditional_losses_4992

inputs1
matmul_readvariableop_resource:	
-
biasadd_readvariableop_resource:

identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
К
j
1__inference_spiking_activation_layer_call_fn_4729

inputs
identityЂStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_spiking_activation_layer_call_and_return_conditional_losses_4045}
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџџџџџџџџџџ22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
љJ
k
L__inference_spiking_activation_layer_call_and_return_conditional_losses_4045

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB T
random_uniform/minConst*
_output_shapes
: *
dtype0*
value	B : X
random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB :џџџџЈ
random_uniformRandomUniformIntrandom_uniform/shape:output:0random_uniform/min:output:0random_uniform/max:output:0*
T0*

Tout0*
_output_shapes
: c
 stateless_random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value
B :
stateless_random_uniform/shapePackstrided_slice:output:0)stateless_random_uniform/shape/1:output:0*
N*
T0*
_output_shapes
:a
stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    a
stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ђ
:stateless_random_uniform/StatelessRandomGetKeyCounter/seedPackrandom_uniform:output:0random_uniform:output:0*
N*
T0*
_output_shapes
:Щ
5stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterCstateless_random_uniform/StatelessRandomGetKeyCounter/seed:output:0*
Tseed0* 
_output_shapes
::w
5stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :о
1stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2'stateless_random_uniform/shape:output:0;stateless_random_uniform/StatelessRandomGetKeyCounter:key:0?stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0>stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*(
_output_shapes
:џџџџџџџџџ
stateless_random_uniform/subSub%stateless_random_uniform/max:output:0%stateless_random_uniform/min:output:0*
T0*
_output_shapes
: Д
stateless_random_uniform/mulMul:stateless_random_uniform/StatelessRandomUniformV2:output:0 stateless_random_uniform/sub:z:0*
T0*(
_output_shapes
:џџџџџџџџџ
stateless_random_uniformAddV2 stateless_random_uniform/mul:z:0%stateless_random_uniform/min:output:0*
T0*(
_output_shapes
:џџџџџџџџџc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ъ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask`
spiking_activation_cell/Cast/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z f
$spiking_activation_cell/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z 
"spiking_activation_cell/LogicalAnd
LogicalAnd'spiking_activation_cell/Cast/x:output:0-spiking_activation_cell/LogicalAnd/y:output:0*
_output_shapes
: q
spiking_activation_cell/ReluRelustrided_slice_2:output:0*
T0*(
_output_shapes
:џџџџџџџџџb
spiking_activation_cell/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:Љ
spiking_activation_cell/mulMul*spiking_activation_cell/Relu:activations:0&spiking_activation_cell/mul/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
spiking_activation_cell/addAddV2stateless_random_uniform:z:0spiking_activation_cell/mul:z:0*
T0*(
_output_shapes
:џџџџџџџџџz
spiking_activation_cell/FloorFloorspiking_activation_cell/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџ
spiking_activation_cell/subSubspiking_activation_cell/add:z:0!spiking_activation_cell/Floor:y:0*
T0*(
_output_shapes
:џџџџџџџџџf
!spiking_activation_cell/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:Ќ
spiking_activation_cell/truedivRealDiv!spiking_activation_cell/Floor:y:0*spiking_activation_cell/truediv/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
 spiking_activation_cell/IdentityIdentity#spiking_activation_cell/truediv:z:0*
T0*(
_output_shapes
:џџџџџџџџџ
"spiking_activation_cell/Identity_1Identityspiking_activation_cell/sub:z:0*
T0*(
_output_shapes
:џџџџџџџџџЭ
!spiking_activation_cell/IdentityN	IdentityN#spiking_activation_cell/truediv:z:0spiking_activation_cell/sub:z:0strided_slice_2:output:0stateless_random_uniform:z:0*
T
2**
_gradient_op_typeCustomGradient-3959*d
_output_shapesR
P:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Р
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0stateless_random_uniform:z:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*
T
	2*
_lower_using_switch_merge(*
_num_original_outputs*4
_output_shapes"
 : : : : :џџџџџџџџџ: : * 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_3978*
condR
while_cond_3977*3
output_shapes"
 : : : : :џџџџџџџџџ: : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ь
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџe
IdentityIdentitytranspose_1:y:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџџџџџџџџџџ:] Y
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
§
Т
__inference__wrapped_model_3538
reshape_inputU
@sequential_time_distributed_dense_matmul_readvariableop_resource:P
Asequential_time_distributed_dense_biasadd_readvariableop_resource:	D
1sequential_dense_1_matmul_readvariableop_resource:	
@
2sequential_dense_1_biasadd_readvariableop_resource:

identityЂ)sequential/dense_1/BiasAdd/ReadVariableOpЂ(sequential/dense_1/MatMul/ReadVariableOpЂ8sequential/time_distributed/dense/BiasAdd/ReadVariableOpЂ7sequential/time_distributed/dense/MatMul/ReadVariableOpU
sequential/reshape/ShapeShapereshape_input*
T0*
_output_shapes
:p
&sequential/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(sequential/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(sequential/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:А
 sequential/reshape/strided_sliceStridedSlice!sequential/reshape/Shape:output:0/sequential/reshape/strided_slice/stack:output:01sequential/reshape/strided_slice/stack_1:output:01sequential/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
"sequential/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџf
"sequential/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
valueB	 :л
 sequential/reshape/Reshape/shapePack)sequential/reshape/strided_slice:output:0+sequential/reshape/Reshape/shape/1:output:0+sequential/reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
: 
sequential/reshape/ReshapeReshapereshape_input)sequential/reshape/Reshape/shape:output:0*
T0*6
_output_shapes$
": џџџџџџџџџџџџџџџџџџt
!sequential/time_distributed/ShapeShape#sequential/reshape/Reshape:output:0*
T0*
_output_shapes
:y
/sequential/time_distributed/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:{
1sequential/time_distributed/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1sequential/time_distributed/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:н
)sequential/time_distributed/strided_sliceStridedSlice*sequential/time_distributed/Shape:output:08sequential/time_distributed/strided_slice/stack:output:0:sequential/time_distributed/strided_slice/stack_1:output:0:sequential/time_distributed/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskz
)sequential/time_distributed/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ @  Л
#sequential/time_distributed/ReshapeReshape#sequential/reshape/Reshape:output:02sequential/time_distributed/Reshape/shape:output:0*
T0*)
_output_shapes
:џџџџџџџџџЛ
7sequential/time_distributed/dense/MatMul/ReadVariableOpReadVariableOp@sequential_time_distributed_dense_matmul_readvariableop_resource*!
_output_shapes
:*
dtype0д
(sequential/time_distributed/dense/MatMulMatMul,sequential/time_distributed/Reshape:output:0?sequential/time_distributed/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЗ
8sequential/time_distributed/dense/BiasAdd/ReadVariableOpReadVariableOpAsequential_time_distributed_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0н
)sequential/time_distributed/dense/BiasAddBiasAdd2sequential/time_distributed/dense/MatMul:product:0@sequential/time_distributed/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџx
-sequential/time_distributed/Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџp
-sequential/time_distributed/Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :
+sequential/time_distributed/Reshape_1/shapePack6sequential/time_distributed/Reshape_1/shape/0:output:02sequential/time_distributed/strided_slice:output:06sequential/time_distributed/Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:к
%sequential/time_distributed/Reshape_1Reshape2sequential/time_distributed/dense/BiasAdd:output:04sequential/time_distributed/Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ|
+sequential/time_distributed/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ @  П
%sequential/time_distributed/Reshape_2Reshape#sequential/reshape/Reshape:output:04sequential/time_distributed/Reshape_2/shape:output:0*
T0*)
_output_shapes
:џџџџџџџџџ
#sequential/spiking_activation/ShapeShape.sequential/time_distributed/Reshape_1:output:0*
T0*
_output_shapes
:{
1sequential/spiking_activation/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3sequential/spiking_activation/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3sequential/spiking_activation/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ч
+sequential/spiking_activation/strided_sliceStridedSlice,sequential/spiking_activation/Shape:output:0:sequential/spiking_activation/strided_slice/stack:output:0<sequential/spiking_activation/strided_slice/stack_1:output:0<sequential/spiking_activation/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
2sequential/spiking_activation/random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB r
0sequential/spiking_activation/random_uniform/minConst*
_output_shapes
: *
dtype0*
value	B : v
0sequential/spiking_activation/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB :џџџџ 
,sequential/spiking_activation/random_uniformRandomUniformInt;sequential/spiking_activation/random_uniform/shape:output:09sequential/spiking_activation/random_uniform/min:output:09sequential/spiking_activation/random_uniform/max:output:0*
T0*

Tout0*
_output_shapes
: 
>sequential/spiking_activation/stateless_random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value
B :ё
<sequential/spiking_activation/stateless_random_uniform/shapePack4sequential/spiking_activation/strided_slice:output:0Gsequential/spiking_activation/stateless_random_uniform/shape/1:output:0*
N*
T0*
_output_shapes
:
:sequential/spiking_activation/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    
:sequential/spiking_activation/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?ќ
Xsequential/spiking_activation/stateless_random_uniform/StatelessRandomGetKeyCounter/seedPack5sequential/spiking_activation/random_uniform:output:05sequential/spiking_activation/random_uniform:output:0*
N*
T0*
_output_shapes
:
Ssequential/spiking_activation/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterasequential/spiking_activation/stateless_random_uniform/StatelessRandomGetKeyCounter/seed:output:0*
Tseed0* 
_output_shapes
::
Ssequential/spiking_activation/stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :є
Osequential/spiking_activation/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Esequential/spiking_activation/stateless_random_uniform/shape:output:0Ysequential/spiking_activation/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0]sequential/spiking_activation/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0\sequential/spiking_activation/stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*(
_output_shapes
:џџџџџџџџџь
:sequential/spiking_activation/stateless_random_uniform/subSubCsequential/spiking_activation/stateless_random_uniform/max:output:0Csequential/spiking_activation/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: 
:sequential/spiking_activation/stateless_random_uniform/mulMulXsequential/spiking_activation/stateless_random_uniform/StatelessRandomUniformV2:output:0>sequential/spiking_activation/stateless_random_uniform/sub:z:0*
T0*(
_output_shapes
:џџџџџџџџџї
6sequential/spiking_activation/stateless_random_uniformAddV2>sequential/spiking_activation/stateless_random_uniform/mul:z:0Csequential/spiking_activation/stateless_random_uniform/min:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
,sequential/spiking_activation/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          л
'sequential/spiking_activation/transpose	Transpose.sequential/time_distributed/Reshape_1:output:05sequential/spiking_activation/transpose/perm:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ
%sequential/spiking_activation/Shape_1Shape+sequential/spiking_activation/transpose:y:0*
T0*
_output_shapes
:}
3sequential/spiking_activation/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5sequential/spiking_activation/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5sequential/spiking_activation/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ё
-sequential/spiking_activation/strided_slice_1StridedSlice.sequential/spiking_activation/Shape_1:output:0<sequential/spiking_activation/strided_slice_1/stack:output:0>sequential/spiking_activation/strided_slice_1/stack_1:output:0>sequential/spiking_activation/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
9sequential/spiking_activation/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
+sequential/spiking_activation/TensorArrayV2TensorListReserveBsequential/spiking_activation/TensorArrayV2/element_shape:output:06sequential/spiking_activation/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвЄ
Ssequential/spiking_activation/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   К
Esequential/spiking_activation/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor+sequential/spiking_activation/transpose:y:0\sequential/spiking_activation/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв}
3sequential/spiking_activation/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5sequential/spiking_activation/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5sequential/spiking_activation/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
-sequential/spiking_activation/strided_slice_2StridedSlice+sequential/spiking_activation/transpose:y:0<sequential/spiking_activation/strided_slice_2/stack:output:0>sequential/spiking_activation/strided_slice_2/stack_1:output:0>sequential/spiking_activation/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask~
<sequential/spiking_activation/spiking_activation_cell/Cast/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 
Bsequential/spiking_activation/spiking_activation_cell/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Zњ
@sequential/spiking_activation/spiking_activation_cell/LogicalAnd
LogicalAndEsequential/spiking_activation/spiking_activation_cell/Cast/x:output:0Ksequential/spiking_activation/spiking_activation_cell/LogicalAnd/y:output:0*
_output_shapes
: ­
:sequential/spiking_activation/spiking_activation_cell/ReluRelu6sequential/spiking_activation/strided_slice_2:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
;sequential/spiking_activation/spiking_activation_cell/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:
9sequential/spiking_activation/spiking_activation_cell/mulMulHsequential/spiking_activation/spiking_activation_cell/Relu:activations:0Dsequential/spiking_activation/spiking_activation_cell/mul/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџ№
9sequential/spiking_activation/spiking_activation_cell/addAddV2:sequential/spiking_activation/stateless_random_uniform:z:0=sequential/spiking_activation/spiking_activation_cell/mul:z:0*
T0*(
_output_shapes
:џџџџџџџџџЖ
;sequential/spiking_activation/spiking_activation_cell/FloorFloor=sequential/spiking_activation/spiking_activation_cell/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџѓ
9sequential/spiking_activation/spiking_activation_cell/subSub=sequential/spiking_activation/spiking_activation_cell/add:z:0?sequential/spiking_activation/spiking_activation_cell/Floor:y:0*
T0*(
_output_shapes
:џџџџџџџџџ
?sequential/spiking_activation/spiking_activation_cell/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:
=sequential/spiking_activation/spiking_activation_cell/truedivRealDiv?sequential/spiking_activation/spiking_activation_cell/Floor:y:0Hsequential/spiking_activation/spiking_activation_cell/truediv/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџР
>sequential/spiking_activation/spiking_activation_cell/IdentityIdentityAsequential/spiking_activation/spiking_activation_cell/truediv:z:0*
T0*(
_output_shapes
:џџџџџџџџџО
@sequential/spiking_activation/spiking_activation_cell/Identity_1Identity=sequential/spiking_activation/spiking_activation_cell/sub:z:0*
T0*(
_output_shapes
:џџџџџџџџџу
?sequential/spiking_activation/spiking_activation_cell/IdentityN	IdentityNAsequential/spiking_activation/spiking_activation_cell/truediv:z:0=sequential/spiking_activation/spiking_activation_cell/sub:z:06sequential/spiking_activation/strided_slice_2:output:0:sequential/spiking_activation/stateless_random_uniform:z:0*
T
2**
_gradient_op_typeCustomGradient-3444*d
_output_shapesR
P:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ
;sequential/spiking_activation/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
-sequential/spiking_activation/TensorArrayV2_1TensorListReserveDsequential/spiking_activation/TensorArrayV2_1/element_shape:output:06sequential/spiking_activation/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвd
"sequential/spiking_activation/timeConst*
_output_shapes
: *
dtype0*
value	B : 
6sequential/spiking_activation/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџr
0sequential/spiking_activation/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ь
#sequential/spiking_activation/whileStatelessWhile9sequential/spiking_activation/while/loop_counter:output:0?sequential/spiking_activation/while/maximum_iterations:output:0+sequential/spiking_activation/time:output:06sequential/spiking_activation/TensorArrayV2_1:handle:0:sequential/spiking_activation/stateless_random_uniform:z:06sequential/spiking_activation/strided_slice_1:output:0Usequential/spiking_activation/TensorArrayUnstack/TensorListFromTensor:output_handle:0*
T
	2*
_lower_using_switch_merge(*
_num_original_outputs*4
_output_shapes"
 : : : : :џџџџџџџџџ: : * 
_read_only_resource_inputs
 *
_stateful_parallelism( *9
body1R/
-sequential_spiking_activation_while_body_3463*9
cond1R/
-sequential_spiking_activation_while_cond_3462*3
output_shapes"
 : : : : :џџџџџџџџџ: : *
parallel_iterations 
Nsequential/spiking_activation/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   І
@sequential/spiking_activation/TensorArrayV2Stack/TensorListStackTensorListStack,sequential/spiking_activation/while:output:3Wsequential/spiking_activation/TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ*
element_dtype0
3sequential/spiking_activation/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџ
5sequential/spiking_activation/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
5sequential/spiking_activation/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
-sequential/spiking_activation/strided_slice_3StridedSliceIsequential/spiking_activation/TensorArrayV2Stack/TensorListStack:tensor:0<sequential/spiking_activation/strided_slice_3/stack:output:0>sequential/spiking_activation/strided_slice_3/stack_1:output:0>sequential/spiking_activation/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask
.sequential/spiking_activation/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          њ
)sequential/spiking_activation/transpose_1	TransposeIsequential/spiking_activation/TensorArrayV2Stack/TensorListStack:tensor:07sequential/spiking_activation/transpose_1/perm:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ|
:sequential/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :з
(sequential/global_average_pooling1d/MeanMean-sequential/spiking_activation/transpose_1:y:0Csequential/global_average_pooling1d/Mean/reduction_indices:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype0К
sequential/dense_1/MatMulMatMul1sequential/global_average_pooling1d/Mean:output:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ

)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0Џ
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
r
IdentityIdentity#sequential/dense_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ

NoOpNoOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp9^sequential/time_distributed/dense/BiasAdd/ReadVariableOp8^sequential/time_distributed/dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:(џџџџџџџџџџџџџџџџџџ: : : : 2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp2t
8sequential/time_distributed/dense/BiasAdd/ReadVariableOp8sequential/time_distributed/dense/BiasAdd/ReadVariableOp2r
7sequential/time_distributed/dense/MatMul/ReadVariableOp7sequential/time_distributed/dense/MatMul/ReadVariableOp:m i
>
_output_shapes,
*:(џџџџџџџџџџџџџџџџџџ
'
_user_specified_namereshape_input
љJ
k
L__inference_spiking_activation_layer_call_and_return_conditional_losses_4862

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB T
random_uniform/minConst*
_output_shapes
: *
dtype0*
value	B : X
random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB :џџџџЈ
random_uniformRandomUniformIntrandom_uniform/shape:output:0random_uniform/min:output:0random_uniform/max:output:0*
T0*

Tout0*
_output_shapes
: c
 stateless_random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value
B :
stateless_random_uniform/shapePackstrided_slice:output:0)stateless_random_uniform/shape/1:output:0*
N*
T0*
_output_shapes
:a
stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    a
stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ђ
:stateless_random_uniform/StatelessRandomGetKeyCounter/seedPackrandom_uniform:output:0random_uniform:output:0*
N*
T0*
_output_shapes
:Щ
5stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterCstateless_random_uniform/StatelessRandomGetKeyCounter/seed:output:0*
Tseed0* 
_output_shapes
::w
5stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :о
1stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2'stateless_random_uniform/shape:output:0;stateless_random_uniform/StatelessRandomGetKeyCounter:key:0?stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0>stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*(
_output_shapes
:џџџџџџџџџ
stateless_random_uniform/subSub%stateless_random_uniform/max:output:0%stateless_random_uniform/min:output:0*
T0*
_output_shapes
: Д
stateless_random_uniform/mulMul:stateless_random_uniform/StatelessRandomUniformV2:output:0 stateless_random_uniform/sub:z:0*
T0*(
_output_shapes
:џџџџџџџџџ
stateless_random_uniformAddV2 stateless_random_uniform/mul:z:0%stateless_random_uniform/min:output:0*
T0*(
_output_shapes
:џџџџџџџџџc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ъ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask`
spiking_activation_cell/Cast/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z f
$spiking_activation_cell/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z 
"spiking_activation_cell/LogicalAnd
LogicalAnd'spiking_activation_cell/Cast/x:output:0-spiking_activation_cell/LogicalAnd/y:output:0*
_output_shapes
: q
spiking_activation_cell/ReluRelustrided_slice_2:output:0*
T0*(
_output_shapes
:џџџџџџџџџb
spiking_activation_cell/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:Љ
spiking_activation_cell/mulMul*spiking_activation_cell/Relu:activations:0&spiking_activation_cell/mul/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
spiking_activation_cell/addAddV2stateless_random_uniform:z:0spiking_activation_cell/mul:z:0*
T0*(
_output_shapes
:џџџџџџџџџz
spiking_activation_cell/FloorFloorspiking_activation_cell/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџ
spiking_activation_cell/subSubspiking_activation_cell/add:z:0!spiking_activation_cell/Floor:y:0*
T0*(
_output_shapes
:џџџџџџџџџf
!spiking_activation_cell/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:Ќ
spiking_activation_cell/truedivRealDiv!spiking_activation_cell/Floor:y:0*spiking_activation_cell/truediv/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
 spiking_activation_cell/IdentityIdentity#spiking_activation_cell/truediv:z:0*
T0*(
_output_shapes
:џџџџџџџџџ
"spiking_activation_cell/Identity_1Identityspiking_activation_cell/sub:z:0*
T0*(
_output_shapes
:џџџџџџџџџЭ
!spiking_activation_cell/IdentityN	IdentityN#spiking_activation_cell/truediv:z:0spiking_activation_cell/sub:z:0strided_slice_2:output:0stateless_random_uniform:z:0*
T
2**
_gradient_op_typeCustomGradient-4776*d
_output_shapesR
P:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Р
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0stateless_random_uniform:z:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*
T
	2*
_lower_using_switch_merge(*
_num_original_outputs*4
_output_shapes"
 : : : : :џџџџџџџџџ: : * 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_4795*
condR
while_cond_4794*3
output_shapes"
 : : : : :џџџџџџџџџ: : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ь
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџe
IdentityIdentitytranspose_1:y:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџџџџџџџџџџ:] Y
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ь
б
)__inference_sequential_layer_call_fn_4331

inputs
unknown:
	unknown_0:	
	unknown_1:	

	unknown_2:

identityЂStatefulPartitionedCallі
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_4237o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:(џџџџџџџџџџџџџџџџџџ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:f b
>
_output_shapes,
*:(џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ь
p
6__inference_spiking_activation_cell_layer_call_fn_5257

inputs
states_0
identity

identity_1р
PartitionedCallPartitionedCallinputsstates_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_spiking_activation_cell_layer_call_and_return_conditional_losses_3686a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџc

Identity_1IdentityPartitionedCall:output:1*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:џџџџџџџџџ:џџџџџџџџџ:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:RN
(
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
states/0
ш
з
!__inference_internal_grad_fn_5373
result_grads_0
result_grads_1
result_grads_2
result_grads_3T
Pones_like_shape_sequential_spiking_activation_while_spiking_activation_cell_relu
identity
ones_like/ShapeShapePones_like_shape_sequential_spiking_activation_while_spiking_activation_cell_relu*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?x
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџї
Rgradient_tape/sequential/spiking_activation/while/spiking_activation_cell/ReluGradReluGradones_like:output:0Pones_like_shape_sequential_spiking_activation_while_spiking_activation_cell_relu*
T0*(
_output_shapes
:џџџџџџџџџ­
mulMul^gradient_tape/sequential/spiking_activation/while/spiking_activation_cell/ReluGrad:backprops:0result_grads_0*
T0*(
_output_shapes
:џџџџџџџџџP
IdentityIdentitymul:z:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*w
_input_shapesf
d:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:X T
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_0:XT
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_1:XT
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_2:XT
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_3:.*
(
_output_shapes
:џџџџџџџџџ
о6
џ
"spiking_activation_while_body_4420B
>spiking_activation_while_spiking_activation_while_loop_counterH
Dspiking_activation_while_spiking_activation_while_maximum_iterations(
$spiking_activation_while_placeholder*
&spiking_activation_while_placeholder_1*
&spiking_activation_while_placeholder_2A
=spiking_activation_while_spiking_activation_strided_slice_1_0}
yspiking_activation_while_tensorarrayv2read_tensorlistgetitem_spiking_activation_tensorarrayunstack_tensorlistfromtensor_0%
!spiking_activation_while_identity'
#spiking_activation_while_identity_1'
#spiking_activation_while_identity_2'
#spiking_activation_while_identity_3'
#spiking_activation_while_identity_4?
;spiking_activation_while_spiking_activation_strided_slice_1{
wspiking_activation_while_tensorarrayv2read_tensorlistgetitem_spiking_activation_tensorarrayunstack_tensorlistfromtensor
Jspiking_activation/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
<spiking_activation/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemyspiking_activation_while_tensorarrayv2read_tensorlistgetitem_spiking_activation_tensorarrayunstack_tensorlistfromtensor_0$spiking_activation_while_placeholderSspiking_activation/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџ*
element_dtype0y
7spiking_activation/while/spiking_activation_cell/Cast/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 
=spiking_activation/while/spiking_activation_cell/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Zы
;spiking_activation/while/spiking_activation_cell/LogicalAnd
LogicalAnd@spiking_activation/while/spiking_activation_cell/Cast/x:output:0Fspiking_activation/while/spiking_activation_cell/LogicalAnd/y:output:0*
_output_shapes
: Е
5spiking_activation/while/spiking_activation_cell/ReluReluCspiking_activation/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*(
_output_shapes
:џџџџџџџџџ{
6spiking_activation/while/spiking_activation_cell/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:є
4spiking_activation/while/spiking_activation_cell/mulMulCspiking_activation/while/spiking_activation_cell/Relu:activations:0?spiking_activation/while/spiking_activation_cell/mul/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџв
4spiking_activation/while/spiking_activation_cell/addAddV2&spiking_activation_while_placeholder_28spiking_activation/while/spiking_activation_cell/mul:z:0*
T0*(
_output_shapes
:џџџџџџџџџЌ
6spiking_activation/while/spiking_activation_cell/FloorFloor8spiking_activation/while/spiking_activation_cell/add:z:0*
T0*(
_output_shapes
:џџџџџџџџџф
4spiking_activation/while/spiking_activation_cell/subSub8spiking_activation/while/spiking_activation_cell/add:z:0:spiking_activation/while/spiking_activation_cell/Floor:y:0*
T0*(
_output_shapes
:џџџџџџџџџ
:spiking_activation/while/spiking_activation_cell/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:ї
8spiking_activation/while/spiking_activation_cell/truedivRealDiv:spiking_activation/while/spiking_activation_cell/Floor:y:0Cspiking_activation/while/spiking_activation_cell/truediv/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџЖ
9spiking_activation/while/spiking_activation_cell/IdentityIdentity<spiking_activation/while/spiking_activation_cell/truediv:z:0*
T0*(
_output_shapes
:џџџџџџџџџД
;spiking_activation/while/spiking_activation_cell/Identity_1Identity8spiking_activation/while/spiking_activation_cell/sub:z:0*
T0*(
_output_shapes
:џџџџџџџџџЭ
:spiking_activation/while/spiking_activation_cell/IdentityN	IdentityN<spiking_activation/while/spiking_activation_cell/truediv:z:08spiking_activation/while/spiking_activation_cell/sub:z:0Cspiking_activation/while/TensorArrayV2Read/TensorListGetItem:item:0&spiking_activation_while_placeholder_2*
T
2**
_gradient_op_typeCustomGradient-4445*d
_output_shapesR
P:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџЅ
=spiking_activation/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem&spiking_activation_while_placeholder_1$spiking_activation_while_placeholderCspiking_activation/while/spiking_activation_cell/IdentityN:output:0*
_output_shapes
: *
element_dtype0:щшв`
spiking_activation/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
spiking_activation/while/addAddV2$spiking_activation_while_placeholder'spiking_activation/while/add/y:output:0*
T0*
_output_shapes
: b
 spiking_activation/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Г
spiking_activation/while/add_1AddV2>spiking_activation_while_spiking_activation_while_loop_counter)spiking_activation/while/add_1/y:output:0*
T0*
_output_shapes
: r
!spiking_activation/while/IdentityIdentity"spiking_activation/while/add_1:z:0*
T0*
_output_shapes
: 
#spiking_activation/while/Identity_1IdentityDspiking_activation_while_spiking_activation_while_maximum_iterations*
T0*
_output_shapes
: r
#spiking_activation/while/Identity_2Identity spiking_activation/while/add:z:0*
T0*
_output_shapes
: В
#spiking_activation/while/Identity_3IdentityMspiking_activation/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: :щшвЇ
#spiking_activation/while/Identity_4IdentityCspiking_activation/while/spiking_activation_cell/IdentityN:output:1*
T0*(
_output_shapes
:џџџџџџџџџ"O
!spiking_activation_while_identity*spiking_activation/while/Identity:output:0"S
#spiking_activation_while_identity_1,spiking_activation/while/Identity_1:output:0"S
#spiking_activation_while_identity_2,spiking_activation/while/Identity_2:output:0"S
#spiking_activation_while_identity_3,spiking_activation/while/Identity_3:output:0"S
#spiking_activation_while_identity_4,spiking_activation/while/Identity_4:output:0"|
;spiking_activation_while_spiking_activation_strided_slice_1=spiking_activation_while_spiking_activation_strided_slice_1_0"є
wspiking_activation_while_tensorarrayv2read_tensorlistgetitem_spiking_activation_tensorarrayunstack_tensorlistfromtensoryspiking_activation_while_tensorarrayv2read_tensorlistgetitem_spiking_activation_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 : : : : :џџџџџџџџџ: : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 
Т
№
while_body_3821
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ї
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:џџџџџџџџџ*
element_dtype0Г
-while/spiking_activation_cell/PartitionedCallPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:џџџџџџџџџ:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_spiking_activation_cell_layer_call_and_return_conditional_losses_3769п
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder6while/spiking_activation_cell/PartitionedCall:output:0*
_output_shapes
: *
element_dtype0:щшвM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: L
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: ]
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: L
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: :щшв
while/Identity_4Identity6while/spiking_activation_cell/PartitionedCall:output:1*
T0*(
_output_shapes
:џџџџџџџџџ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"Ј
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 : : : : :џџџџџџџџџ: : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
: 

Х
J__inference_time_distributed_layer_call_and_return_conditional_losses_3612

inputs

dense_3602:

dense_3604:	
identityЂdense/StatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ @  f
ReshapeReshapeinputsReshape/shape:output:0*
T0*)
_output_shapes
:џџџџџџџџџь
dense/StatefulPartitionedCallStatefulPartitionedCallReshape:output:0
dense_3602
dense_3604*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_3562\
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
Reshape_1/shape/2Const*
_output_shapes
: *
dtype0*
value
B :
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice:output:0Reshape_1/shape/2:output:0*
N*
T0*
_output_shapes
:
	Reshape_1Reshape&dense/StatefulPartitionedCall:output:0Reshape_1/shape:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџo
IdentityIdentityReshape_1:output:0^NoOp*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџf
NoOpNoOp^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&: џџџџџџџџџџџџџџџџџџ: : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:^ Z
6
_output_shapes$
": џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ђ
]
A__inference_reshape_layer_call_and_return_conditional_losses_3908

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџS
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
valueB	 :
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:s
ReshapeReshapeinputsReshape/shape:output:0*
T0*6
_output_shapes$
": џџџџџџџџџџџџџџџџџџg
IdentityIdentityReshape:output:0*
T0*6
_output_shapes$
": џџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:(џџџџџџџџџџџџџџџџџџ:f b
>
_output_shapes,
*:(џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Х>
k
L__inference_spiking_activation_layer_call_and_return_conditional_losses_4962

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB T
random_uniform/minConst*
_output_shapes
: *
dtype0*
value	B : X
random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB :џџџџЈ
random_uniformRandomUniformIntrandom_uniform/shape:output:0random_uniform/min:output:0random_uniform/max:output:0*
T0*

Tout0*
_output_shapes
: c
 stateless_random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value
B :
stateless_random_uniform/shapePackstrided_slice:output:0)stateless_random_uniform/shape/1:output:0*
N*
T0*
_output_shapes
:a
stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    a
stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ђ
:stateless_random_uniform/StatelessRandomGetKeyCounter/seedPackrandom_uniform:output:0random_uniform:output:0*
N*
T0*
_output_shapes
:Щ
5stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterCstateless_random_uniform/StatelessRandomGetKeyCounter/seed:output:0*
Tseed0* 
_output_shapes
::w
5stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :о
1stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2'stateless_random_uniform/shape:output:0;stateless_random_uniform/StatelessRandomGetKeyCounter:key:0?stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0>stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*(
_output_shapes
:џџџџџџџџџ
stateless_random_uniform/subSub%stateless_random_uniform/max:output:0%stateless_random_uniform/min:output:0*
T0*
_output_shapes
: Д
stateless_random_uniform/mulMul:stateless_random_uniform/StatelessRandomUniformV2:output:0 stateless_random_uniform/sub:z:0*
T0*(
_output_shapes
:џџџџџџџџџ
stateless_random_uniformAddV2 stateless_random_uniform/mul:z:0%stateless_random_uniform/min:output:0*
T0*(
_output_shapes
:џџџџџџџџџc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ъ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask`
spiking_activation_cell/Cast/xConst*
_output_shapes
: *
dtype0
*
value	B
 Zf
$spiking_activation_cell/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z 
"spiking_activation_cell/LogicalAnd
LogicalAnd'spiking_activation_cell/Cast/x:output:0-spiking_activation_cell/LogicalAnd/y:output:0*
_output_shapes
: q
spiking_activation_cell/ReluRelustrided_slice_2:output:0*
T0*(
_output_shapes
:џџџџџџџџџn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Р
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0stateless_random_uniform:z:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*
T
	2*
_lower_using_switch_merge(*
_num_original_outputs*4
_output_shapes"
 : : : : :џџџџџџџџџ: : * 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_4909*
condR
while_cond_4908*3
output_shapes"
 : : : : :џџџџџџџџџ: : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ь
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџe
IdentityIdentitytranspose_1:y:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџџџџџџџџџџ:] Y
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Х

$__inference_dense_layer_call_fn_5001

inputs
unknown:
	unknown_0:	
identityЂStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_3562p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Х>
k
L__inference_spiking_activation_layer_call_and_return_conditional_losses_4194

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB T
random_uniform/minConst*
_output_shapes
: *
dtype0*
value	B : X
random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB :џџџџЈ
random_uniformRandomUniformIntrandom_uniform/shape:output:0random_uniform/min:output:0random_uniform/max:output:0*
T0*

Tout0*
_output_shapes
: c
 stateless_random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value
B :
stateless_random_uniform/shapePackstrided_slice:output:0)stateless_random_uniform/shape/1:output:0*
N*
T0*
_output_shapes
:a
stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    a
stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Ђ
:stateless_random_uniform/StatelessRandomGetKeyCounter/seedPackrandom_uniform:output:0random_uniform:output:0*
N*
T0*
_output_shapes
:Щ
5stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterCstateless_random_uniform/StatelessRandomGetKeyCounter/seed:output:0*
Tseed0* 
_output_shapes
::w
5stateless_random_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :о
1stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2'stateless_random_uniform/shape:output:0;stateless_random_uniform/StatelessRandomGetKeyCounter:key:0?stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0>stateless_random_uniform/StatelessRandomUniformV2/alg:output:0*(
_output_shapes
:џџџџџџџџџ
stateless_random_uniform/subSub%stateless_random_uniform/max:output:0%stateless_random_uniform/min:output:0*
T0*
_output_shapes
: Д
stateless_random_uniform/mulMul:stateless_random_uniform/StatelessRandomUniformV2:output:0 stateless_random_uniform/sub:z:0*
T0*(
_output_shapes
:џџџџџџџџџ
stateless_random_uniformAddV2 stateless_random_uniform/mul:z:0%stateless_random_uniform/min:output:0*
T0*(
_output_shapes
:џџџџџџџџџc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          w
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџД
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   р
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшв_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ъ
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_mask`
spiking_activation_cell/Cast/xConst*
_output_shapes
: *
dtype0
*
value	B
 Zf
$spiking_activation_cell/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z 
"spiking_activation_cell/LogicalAnd
LogicalAnd'spiking_activation_cell/Cast/x:output:0-spiking_activation_cell/LogicalAnd/y:output:0*
_output_shapes
: q
spiking_activation_cell/ReluRelustrided_slice_2:output:0*
T0*(
_output_shapes
:џџџџџџџџџn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:щшвF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Р
whileStatelessWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0stateless_random_uniform:z:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*
T
	2*
_lower_using_switch_merge(*
_num_original_outputs*4
_output_shapes"
 : : : : :џџџџџџџџџ: : * 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
while_body_4141*
condR
while_cond_4140*3
output_shapes"
 : : : : :џџџџџџџџџ: : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ь
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:џџџџџџџџџ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"           
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџe
IdentityIdentitytranspose_1:y:0*
T0*5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:џџџџџџџџџџџџџџџџџџ:] Y
5
_output_shapes#
!:џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Й


!__inference_internal_grad_fn_5513
result_grads_0
result_grads_1
result_grads_2
result_grads_3
ones_like_shape_relu
identityS
ones_like/ShapeShapeones_like_shape_relu*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?x
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
gradient_tape/ReluGradReluGradones_like:output:0ones_like_shape_relu*
T0*(
_output_shapes
:џџџџџџџџџq
mulMul"gradient_tape/ReluGrad:backprops:0result_grads_0*
T0*(
_output_shapes
:џџџџџџџџџP
IdentityIdentitymul:z:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*w
_input_shapesf
d:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:X T
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_0:XT
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_1:XT
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_2:XT
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_3:.*
(
_output_shapes
:џџџџџџџџџ
Г
Г
!__inference_internal_grad_fn_5387
result_grads_0
result_grads_1
result_grads_2
result_grads_30
,ones_like_shape_spiking_activation_cell_relu
identityk
ones_like/ShapeShape,ones_like_shape_spiking_activation_cell_relu*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?x
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџЏ
.gradient_tape/spiking_activation_cell/ReluGradReluGradones_like:output:0,ones_like_shape_spiking_activation_cell_relu*
T0*(
_output_shapes
:џџџџџџџџџ
mulMul:gradient_tape/spiking_activation_cell/ReluGrad:backprops:0result_grads_0*
T0*(
_output_shapes
:џџџџџџџџџP
IdentityIdentitymul:z:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*w
_input_shapesf
d:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:X T
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_0:XT
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_1:XT
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_2:XT
(
_output_shapes
:џџџџџџџџџ
(
_user_specified_nameresult_grads_3:.*
(
_output_shapes
:џџџџџџџџџ
с
и
)__inference_sequential_layer_call_fn_4261
reshape_input
unknown:
	unknown_0:	
	unknown_1:	

	unknown_2:

identityЂStatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallreshape_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ
*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_4237o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:(џџџџџџџџџџџџџџџџџџ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:m i
>
_output_shapes,
*:(џџџџџџџџџџџџџџџџџџ
'
_user_specified_namereshape_input
Њ

while_cond_3820
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_12
.while_while_cond_3820___redundant_placeholder0
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*5
_input_shapes$
": : : : :џџџџџџџџџ: :: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:џџџџџџџџџ:

_output_shapes
: :

_output_shapes
:8
!__inference_internal_grad_fn_5359CustomGradient-34448
!__inference_internal_grad_fn_5373CustomGradient-34888
!__inference_internal_grad_fn_5387CustomGradient-39598
!__inference_internal_grad_fn_5401CustomGradient-40038
!__inference_internal_grad_fn_5415CustomGradient-44018
!__inference_internal_grad_fn_5429CustomGradient-44458
!__inference_internal_grad_fn_5443CustomGradient-47768
!__inference_internal_grad_fn_5457CustomGradient-48208
!__inference_internal_grad_fn_5471CustomGradient-36698
!__inference_internal_grad_fn_5485CustomGradient-50638
!__inference_internal_grad_fn_5499CustomGradient-51078
!__inference_internal_grad_fn_5513CustomGradient-5272"лL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Э
serving_defaultЙ
^
reshape_inputM
serving_default_reshape_input:0(џџџџџџџџџџџџџџџџџџ;
dense_10
StatefulPartitionedCall:0џџџџџџџџџ
tensorflow/serving/predict:цЦ
л
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
	optimizer
	variables
trainable_variables
	regularization_losses

	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_sequential
Ѕ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
А
	layer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
А
	layer
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses"
_tf_keras_layer
Ѕ
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses"
_tf_keras_layer
Л

)kernel
*bias
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses"
_tf_keras_layer
Ѓ
1iter

2beta_1

3beta_2
	4decay
5learning_rate)m*m6m7m)v*v6v7v"
	optimizer
<
60
71
)2
*3"
trackable_list_wrapper
<
60
71
)2
*3"
trackable_list_wrapper
 "
trackable_list_wrapper
Ъ
8non_trainable_variables

9layers
:metrics
;layer_regularization_losses
<layer_metrics
	variables
trainable_variables
	regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ђ2я
)__inference_sequential_layer_call_fn_4076
)__inference_sequential_layer_call_fn_4318
)__inference_sequential_layer_call_fn_4331
)__inference_sequential_layer_call_fn_4261Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
о2л
D__inference_sequential_layer_call_and_return_conditional_losses_4495
D__inference_sequential_layer_call_and_return_conditional_losses_4631
D__inference_sequential_layer_call_and_return_conditional_losses_4280
D__inference_sequential_layer_call_and_return_conditional_losses_4299Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
аBЭ
__inference__wrapped_model_3538reshape_input"
В
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
,
=serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
а2Э
&__inference_reshape_layer_call_fn_4651Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ы2ш
A__inference_reshape_layer_call_and_return_conditional_losses_4664Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Л

6kernel
7bias
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses"
_tf_keras_layer
.
60
71"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Inon_trainable_variables

Jlayers
Kmetrics
Llayer_regularization_losses
Mlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ј2Ѕ
/__inference_time_distributed_layer_call_fn_4673
/__inference_time_distributed_layer_call_fn_4682Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
о2л
J__inference_time_distributed_layer_call_and_return_conditional_losses_4703
J__inference_time_distributed_layer_call_and_return_conditional_losses_4724Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
У
Ncell
O
state_spec
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
Vnon_trainable_variables

Wlayers
Xmetrics
Ylayer_regularization_losses
Zlayer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
Ц2У
1__inference_spiking_activation_layer_call_fn_4729
1__inference_spiking_activation_layer_call_fn_4734к
бВЭ
FullArgSpecG
args?<
jself
jinputs

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults
p 

 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ќ2љ
L__inference_spiking_activation_layer_call_and_return_conditional_losses_4862
L__inference_spiking_activation_layer_call_and_return_conditional_losses_4962к
бВЭ
FullArgSpecG
args?<
jself
jinputs

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults
p 

 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses"
_generic_user_object
ю2ы
7__inference_global_average_pooling1d_layer_call_fn_4967Џ
ІВЂ
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaultsЂ

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_4973Џ
ІВЂ
FullArgSpec%
args
jself
jinputs
jmask
varargs
 
varkw
 
defaultsЂ

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
!:	
2dense_1/kernel
:
2dense_1/bias
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
`non_trainable_variables

alayers
bmetrics
clayer_regularization_losses
dlayer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses"
_generic_user_object
а2Э
&__inference_dense_1_layer_call_fn_4982Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ы2ш
A__inference_dense_1_layer_call_and_return_conditional_losses_4992Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
,:*2time_distributed/kernel
$:"2time_distributed/bias
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
.
e0
f1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЯBЬ
"__inference_signature_wrapper_4646reshape_input"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
60
71"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
­
gnon_trainable_variables

hlayers
imetrics
jlayer_regularization_losses
klayer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
Ю2Ы
$__inference_dense_layer_call_fn_5001Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
щ2ц
?__inference_dense_layer_call_and_return_conditional_losses_5011Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
Ѕ
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
p__call__
*q&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Й

rstates
snon_trainable_variables

tlayers
umetrics
vlayer_regularization_losses
wlayer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
Д2Б
"__inference_rnn_layer_call_fn_5016
"__inference_rnn_layer_call_fn_5021ц
нВй
FullArgSpecO
argsGD
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults

 
p 

 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ъ2ч
=__inference_rnn_layer_call_and_return_conditional_losses_5149
=__inference_rnn_layer_call_and_return_conditional_losses_5249ц
нВй
FullArgSpecO
argsGD
jself
jinputs
jmask

jtraining
jinitial_state
j	constants
varargs
 
varkw
 
defaults

 
p 

 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
N
	xtotal
	ycount
z	variables
{	keras_api"
_tf_keras_metric
_
	|total
	}count
~
_fn_kwargs
	variables
	keras_api"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
l	variables
mtrainable_variables
nregularization_losses
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses"
_generic_user_object
Д2Б
6__inference_spiking_activation_cell_layer_call_fn_5257
6__inference_spiking_activation_cell_layer_call_fn_5265О
ЕВБ
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ъ2ч
Q__inference_spiking_activation_cell_layer_call_and_return_conditional_losses_5289
Q__inference_spiking_activation_cell_layer_call_and_return_conditional_losses_5299О
ЕВБ
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
N0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
.
x0
y1"
trackable_list_wrapper
-
z	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
|0
}1"
trackable_list_wrapper
-
	variables"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
&:$	
2Adam/dense_1/kernel/m
:
2Adam/dense_1/bias/m
1:/2Adam/time_distributed/kernel/m
):'2Adam/time_distributed/bias/m
&:$	
2Adam/dense_1/kernel/v
:
2Adam/dense_1/bias/v
1:/2Adam/time_distributed/kernel/v
):'2Adam/time_distributed/bias/v
ab_
<sequential/spiking_activation/spiking_activation_cell/Relu:0__inference__wrapped_model_3538
ubs
Bsequential/spiking_activation/while/spiking_activation_cell/Relu:0-sequential_spiking_activation_while_body_3463
pbn
spiking_activation_cell/Relu:0L__inference_spiking_activation_layer_call_and_return_conditional_losses_4045
9b7
$while/spiking_activation_cell/Relu:0while_body_3978
{by
1spiking_activation/spiking_activation_cell/Relu:0D__inference_sequential_layer_call_and_return_conditional_losses_4495
_b]
7spiking_activation/while/spiking_activation_cell/Relu:0"spiking_activation_while_body_4420
pbn
spiking_activation_cell/Relu:0L__inference_spiking_activation_layer_call_and_return_conditional_losses_4862
9b7
$while/spiking_activation_cell/Relu:0while_body_4795
]b[
Relu:0Q__inference_spiking_activation_cell_layer_call_and_return_conditional_losses_3686
ab_
spiking_activation_cell/Relu:0=__inference_rnn_layer_call_and_return_conditional_losses_5149
9b7
$while/spiking_activation_cell/Relu:0while_body_5082
]b[
Relu:0Q__inference_spiking_activation_cell_layer_call_and_return_conditional_losses_5289Ќ
__inference__wrapped_model_353867)*MЂJ
CЂ@
>;
reshape_input(џџџџџџџџџџџџџџџџџџ
Њ "1Њ.
,
dense_1!
dense_1џџџџџџџџџ
Ђ
A__inference_dense_1_layer_call_and_return_conditional_losses_4992])*0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ

 z
&__inference_dense_1_layer_call_fn_4982P)*0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "џџџџџџџџџ
Ђ
?__inference_dense_layer_call_and_return_conditional_losses_5011_671Ђ.
'Ђ$
"
inputsџџџџџџџџџ
Њ "&Ђ#

0џџџџџџџџџ
 z
$__inference_dense_layer_call_fn_5001R671Ђ.
'Ђ$
"
inputsџџџџџџџџџ
Њ "џџџџџџџџџб
R__inference_global_average_pooling1d_layer_call_and_return_conditional_losses_4973{IЂF
?Ђ<
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ

 
Њ ".Ђ+
$!
0џџџџџџџџџџџџџџџџџџ
 Љ
7__inference_global_average_pooling1d_layer_call_fn_4967nIЂF
?Ђ<
63
inputs'џџџџџџџџџџџџџџџџџџџџџџџџџџџ

 
Њ "!џџџџџџџџџџџџџџџџџџ
!__inference_internal_grad_fn_5359іРЂМ
ДЂА

 
)&
result_grads_0џџџџџџџџџ
)&
result_grads_1џџџџџџџџџ
)&
result_grads_2џџџџџџџџџ
)&
result_grads_3џџџџџџџџџ
Њ "-*

 

 

2џџџџџџџџџ

 
!__inference_internal_grad_fn_5373іРЂМ
ДЂА

 
)&
result_grads_0џџџџџџџџџ
)&
result_grads_1џџџџџџџџџ
)&
result_grads_2џџџџџџџџџ
)&
result_grads_3џџџџџџџџџ
Њ "-*

 

 

2џџџџџџџџџ

 
!__inference_internal_grad_fn_5387іРЂМ
ДЂА

 
)&
result_grads_0џџџџџџџџџ
)&
result_grads_1џџџџџџџџџ
)&
result_grads_2џџџџџџџџџ
)&
result_grads_3џџџџџџџџџ
Њ "-*

 

 

2џџџџџџџџџ

 
!__inference_internal_grad_fn_5401іРЂМ
ДЂА

 
)&
result_grads_0џџџџџџџџџ
)&
result_grads_1џџџџџџџџџ
)&
result_grads_2џџџџџџџџџ
)&
result_grads_3џџџџџџџџџ
Њ "-*

 

 

2џџџџџџџџџ

 
!__inference_internal_grad_fn_5415іРЂМ
ДЂА

 
)&
result_grads_0џџџџџџџџџ
)&
result_grads_1џџџџџџџџџ
)&
result_grads_2џџџџџџџџџ
)&
result_grads_3џџџџџџџџџ
Њ "-*

 

 

2џџџџџџџџџ

 
!__inference_internal_grad_fn_5429іРЂМ
ДЂА

 
)&
result_grads_0џџџџџџџџџ
)&
result_grads_1џџџџџџџџџ
)&
result_grads_2џџџџџџџџџ
)&
result_grads_3џџџџџџџџџ
Њ "-*

 

 

2џџџџџџџџџ

 
!__inference_internal_grad_fn_5443іРЂМ
ДЂА

 
)&
result_grads_0џџџџџџџџџ
)&
result_grads_1џџџџџџџџџ
)&
result_grads_2џџџџџџџџџ
)&
result_grads_3џџџџџџџџџ
Њ "-*

 

 

2џџџџџџџџџ

 
!__inference_internal_grad_fn_5457іРЂМ
ДЂА

 
)&
result_grads_0џџџџџџџџџ
)&
result_grads_1џџџџџџџџџ
)&
result_grads_2џџџџџџџџџ
)&
result_grads_3џџџџџџџџџ
Њ "-*

 

 

2џџџџџџџџџ

 
!__inference_internal_grad_fn_5471іРЂМ
ДЂА

 
)&
result_grads_0џџџџџџџџџ
)&
result_grads_1џџџџџџџџџ
)&
result_grads_2џџџџџџџџџ
)&
result_grads_3џџџџџџџџџ
Њ "-*

 

 

2џџџџџџџџџ

 
!__inference_internal_grad_fn_5485іРЂМ
ДЂА

 
)&
result_grads_0џџџџџџџџџ
)&
result_grads_1џџџџџџџџџ
)&
result_grads_2џџџџџџџџџ
)&
result_grads_3џџџџџџџџџ
Њ "-*

 

 

2џџџџџџџџџ

 
!__inference_internal_grad_fn_5499іРЂМ
ДЂА

 
)&
result_grads_0џџџџџџџџџ
)&
result_grads_1џџџџџџџџџ
)&
result_grads_2џџџџџџџџџ
)&
result_grads_3џџџџџџџџџ
Њ "-*

 

 

2џџџџџџџџџ

 
!__inference_internal_grad_fn_5513іРЂМ
ДЂА

 
)&
result_grads_0џџџџџџџџџ
)&
result_grads_1џџџџџџџџџ
)&
result_grads_2џџџџџџџџџ
)&
result_grads_3џџџџџџџџџ
Њ "-*

 

 

2џџџџџџџџџ

 У
A__inference_reshape_layer_call_and_return_conditional_losses_4664~FЂC
<Ђ9
74
inputs(џџџџџџџџџџџџџџџџџџ
Њ "4Ђ1
*'
0 џџџџџџџџџџџџџџџџџџ
 
&__inference_reshape_layer_call_fn_4651qFЂC
<Ђ9
74
inputs(џџџџџџџџџџџџџџџџџџ
Њ "'$ џџџџџџџџџџџџџџџџџџЭ
=__inference_rnn_layer_call_and_return_conditional_losses_5149TЂQ
JЂG
52
0-
inputs/0џџџџџџџџџџџџџџџџџџ

 
p 

 

 
Њ "3Ђ0
)&
0џџџџџџџџџџџџџџџџџџ
 Э
=__inference_rnn_layer_call_and_return_conditional_losses_5249TЂQ
JЂG
52
0-
inputs/0џџџџџџџџџџџџџџџџџџ

 
p

 

 
Њ "3Ђ0
)&
0џџџџџџџџџџџџџџџџџџ
 Є
"__inference_rnn_layer_call_fn_5016~TЂQ
JЂG
52
0-
inputs/0џџџџџџџџџџџџџџџџџџ

 
p 

 

 
Њ "&#џџџџџџџџџџџџџџџџџџЄ
"__inference_rnn_layer_call_fn_5021~TЂQ
JЂG
52
0-
inputs/0џџџџџџџџџџџџџџџџџџ

 
p

 

 
Њ "&#џџџџџџџџџџџџџџџџџџЭ
D__inference_sequential_layer_call_and_return_conditional_losses_428067)*UЂR
KЂH
>;
reshape_input(џџџџџџџџџџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ

 Э
D__inference_sequential_layer_call_and_return_conditional_losses_429967)*UЂR
KЂH
>;
reshape_input(џџџџџџџџџџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ

 Х
D__inference_sequential_layer_call_and_return_conditional_losses_4495}67)*NЂK
DЂA
74
inputs(џџџџџџџџџџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ

 Х
D__inference_sequential_layer_call_and_return_conditional_losses_4631}67)*NЂK
DЂA
74
inputs(џџџџџџџџџџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ

 Є
)__inference_sequential_layer_call_fn_4076w67)*UЂR
KЂH
>;
reshape_input(џџџџџџџџџџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
Є
)__inference_sequential_layer_call_fn_4261w67)*UЂR
KЂH
>;
reshape_input(џџџџџџџџџџџџџџџџџџ
p

 
Њ "џџџџџџџџџ

)__inference_sequential_layer_call_fn_4318p67)*NЂK
DЂA
74
inputs(џџџџџџџџџџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ

)__inference_sequential_layer_call_fn_4331p67)*NЂK
DЂA
74
inputs(џџџџџџџџџџџџџџџџџџ
p

 
Њ "џџџџџџџџџ
Р
"__inference_signature_wrapper_464667)*^Ђ[
Ђ 
TЊQ
O
reshape_input>;
reshape_input(џџџџџџџџџџџџџџџџџџ"1Њ.
,
dense_1!
dense_1џџџџџџџџџ

Q__inference_spiking_activation_cell_layer_call_and_return_conditional_losses_5289Ж^Ђ[
TЂQ
!
inputsџџџџџџџџџ
(Ђ%
# 
states/0џџџџџџџџџ
p 
Њ "TЂQ
JЂG

0/0џџџџџџџџџ
%Ђ"
 
0/1/0џџџџџџџџџ
 
Q__inference_spiking_activation_cell_layer_call_and_return_conditional_losses_5299Ж^Ђ[
TЂQ
!
inputsџџџџџџџџџ
(Ђ%
# 
states/0џџџџџџџџџ
p
Њ "TЂQ
JЂG

0/0џџџџџџџџџ
%Ђ"
 
0/1/0џџџџџџџџџ
 у
6__inference_spiking_activation_cell_layer_call_fn_5257Ј^Ђ[
TЂQ
!
inputsџџџџџџџџџ
(Ђ%
# 
states/0џџџџџџџџџ
p 
Њ "FЂC

0џџџџџџџџџ
#Ђ 

1/0џџџџџџџџџу
6__inference_spiking_activation_cell_layer_call_fn_5265Ј^Ђ[
TЂQ
!
inputsџџџџџџџџџ
(Ђ%
# 
states/0џџџџџџџџџ
p
Њ "FЂC

0џџџџџџџџџ
#Ђ 

1/0џџџџџџџџџб
L__inference_spiking_activation_layer_call_and_return_conditional_losses_4862IЂF
?Ђ<
.+
inputsџџџџџџџџџџџџџџџџџџ
p 

 

 
Њ "3Ђ0
)&
0џџџџџџџџџџџџџџџџџџ
 б
L__inference_spiking_activation_layer_call_and_return_conditional_losses_4962IЂF
?Ђ<
.+
inputsџџџџџџџџџџџџџџџџџџ
p

 

 
Њ "3Ђ0
)&
0џџџџџџџџџџџџџџџџџџ
 Ј
1__inference_spiking_activation_layer_call_fn_4729sIЂF
?Ђ<
.+
inputsџџџџџџџџџџџџџџџџџџ
p 

 

 
Њ "&#џџџџџџџџџџџџџџџџџџЈ
1__inference_spiking_activation_layer_call_fn_4734sIЂF
?Ђ<
.+
inputsџџџџџџџџџџџџџџџџџџ
p

 

 
Њ "&#џџџџџџџџџџџџџџџџџџа
J__inference_time_distributed_layer_call_and_return_conditional_losses_470367FЂC
<Ђ9
/,
inputs џџџџџџџџџџџџџџџџџџ
p 

 
Њ "3Ђ0
)&
0џџџџџџџџџџџџџџџџџџ
 а
J__inference_time_distributed_layer_call_and_return_conditional_losses_472467FЂC
<Ђ9
/,
inputs џџџџџџџџџџџџџџџџџџ
p

 
Њ "3Ђ0
)&
0џџџџџџџџџџџџџџџџџџ
 Ї
/__inference_time_distributed_layer_call_fn_4673t67FЂC
<Ђ9
/,
inputs џџџџџџџџџџџџџџџџџџ
p 

 
Њ "&#џџџџџџџџџџџџџџџџџџЇ
/__inference_time_distributed_layer_call_fn_4682t67FЂC
<Ђ9
/,
inputs џџџџџџџџџџџџџџџџџџ
p

 
Њ "&#џџџџџџџџџџџџџџџџџџ