эЈ9
В$÷#
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
P
Assert
	condition
	
data2T"
T
list(type)(0"
	summarizeintИ
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
p
BatchMatMulV2
x"T
y"T
output"T"
Ttype:
2	"
adj_xbool( "
adj_ybool( 
А
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
Н
DenseBincount
input"Tidx
size"Tidx
weights"T
output"T"
Tidxtype:
2	"
Ttype:
2	"
binary_outputbool( 
$
DisableCopyOnRead
resourceИ
R
Equal
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(Р
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
q
GatherNd
params"Tparams
indices"Tindices
output"Tparams"
Tparamstype"
Tindicestype:
2	
=
Greater
x"T
y"T
z
"
Ttype:
2	
°
HashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetypeИ
.
Identity

input"T
output"T"	
Ttype
w
LookupTableFindV2
table_handle
keys"Tin
default_value"Tout
values"Tout"
Tintype"
TouttypeИ
b
LookupTableImportV2
table_handle
keys"Tin
values"Tout"
Tintype"
TouttypeИ
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
М
Max

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
Ж
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( И
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	Р
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
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
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
•
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	И
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
d
Shape

input"T&
output"out_typeКнout_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
O
Size

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
7
Square
x"T
y"T"
Ttype:
2	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
Ѕ
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
executor_typestring И®
@
StaticRegexFullMatch	
input

output
"
patternstring
З
StringFormat
inputs2T

output"
T
list(type)("
templatestring%s"
placeholderstring%s"
	summarizeint
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
О
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( ""
Ttype:
2	"
Tidxtype0:
2	
∞
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И
G
Where

input"T	
index	"'
Ttype0
:
2	
"serve*2.15.02v2.15.0-0-g6887368d6d48РЮ.
м	
ConstConst*
_output_shapes	
:У*
dtype0	*±	
valueІ	B§		У"Ш	                                                                	       
                                                                                                                                                                  !       "       #       $       %       &       '       (       )       *       +       ,       -       .       /       0       1       2       3       4       5       6       7       8       9       :       ;       <       =       >       ?       @       A       B       C       D       E       F       G       H       I       J       K       L       M       N       O       P       Q       R       S       T       U       V       W       X       Y       Z       [       \       ]       ^       _       `       a       b       c       d       e       f       g       h       i       j       k       l       m       n       o       p       q       r       s       t       u       v       w       x       y       z       {       |       }       ~              А       Б       В       Г       Д       Е       Ж       З       И       Й       К       Л       М       Н       О       П       Р       С       Т       
М
Const_1Const*
_output_shapes	
:У*
dtype0*ѕ
value≈B¬УBveggieB
Meat-basedBveganB
vegetarianBsmoothieBgrain-basedBdessertB
VegetarianBlunch;dinnerB
meat-basedBNotInformationBBeverageBfruit-basedBDessertBpastaBcheesyB	breakfastBitalianBbreakfast;lunch;dinnerBStuffingB	AppetizerBseafoodBsushiB,the meal type for this recipe is vegetarian.BFruit-basedBfishB/this recipe can be classified as seafood-based.Bbreakfast;dinnerBUbased on the ingredients provided, the meal type of the recipe would be "vegetarian."B*The meal type for this recipe is "veggie".BMexicanBSeafoodBMuffinsBbrunchBBasicBsnackBindianBsandwichBasianBprotein-basedB	appetizerBcheeseBbreakfast;lunchBDrinkBbreadB+The meal type of the recipe is grain-based.BdinnerBlunchB,in this recipe, the meal type is vegetarian.BGrain-basedB
Meat-basedBsaladBBiscuitsBmexicanBBread-basedBSnackBVeggieB	BreakfastB,the meal type for this recipe is meat-based.BfruitB+the meal type of this recipe is meat-based.BBakingBVeganBSpice-basedB	CondimentBBrunchB
Herb-basedBbaked goodsBlactose-freeBKosherBBakedBђno information is given about the specific dietary restrictions or preferences, but based on the ingredients listed, the meal type for this recipe would be "protein-based."BSeafood-basedBЈsorry, i am unable to generate the recipe based on the information provided. however, the meal type of the cheese stuffed mini bell peppers would typically be considered "vegetarian."BThis recipe is meat-based.Blebanese fatayer- meat-basedBveggie/vegetarianBpescatarianBJthere is not enough information to determine the meal type of this recipe.BmeatlessB*the meal type of this recipe is "dessert".BbeverageBoatmealBnon-vegetarianBpizzaB
fish-basedBeggsB	sandwich.BdairyBthis recipe is meat-based.BbakingB	egg-basedBCandyBvegetarian.BBaked-GoodsBbaklavaB	nut-basedBsweetBDairyBCocktailBcapreseBtoastBomeletteB)the meal type for this recipe is seafood.B	AlcoholicBSpicyBkoreanB+the meal type of this recipe is vegetarian.Bplant-basedBcoconutB
bruschettaBfusionBthe meal type is greek.BkosherBhalalBsmoothie bowlBPicklesB&the meal type of the recipe is indian.B^keywords like "veggie" and "hummus" indicate that the meal type for this recipe is vegetarian.Bnot specifiedBcheese-basedB plain, chocolate chip, raspberryBthis recipe is vegetarian.BpastryBunspecifiedBgreekBbakeryBbreadsB(the meal type of the recipe is "pastry".BspicyBfrozen dessertBsavoryB!berry-yogurt parfait - vegetarianBIthe meal types for the provided recipes are seafood-based or sushi-based.Bpasta.B'the meal type of the recipe is "vegan".BRbased on the ingredients listed, the meal type of this recipe would be vegetarian.BmeatBamericanB'the meal type for this recipe is sushi.BRas the recipe includes both tofu or chicken, it can be considered as "meat-based".B	graubasedBProteinB#this recipe is a meat-based recipe.BSweetBEthere is no specific indication of the meal type in the given recipe.B)the meal type of the recipe is "dessert."
№
Const_2Const*
_output_shapes
:*
dtype0	*†
valueЦBУ	"И                                                                	       
                                                 
Ћ
Const_3Const*
_output_shapes
:*
dtype0*П
valueЕBВBsaltyBsweetBsourBbitterBumamiBsavoryBumami, saltyBumami, sourBbitter, umami, saltyBbitter, sweet, saltyBsour, saltyBumami, sour, saltyB!umami, sour, sweet, bitter, saltyBsweet, sourBsweet, saltyBumami, sweet, sour, saltyBsweet, umami
p
Const_4Const*
_output_shapes
:*
dtype0	*5
value,B*	"                              
r
Const_5Const*
_output_shapes
:*
dtype0*7
value.B,BaloneB
colleaguesBfriendsBfamily
i
Const_6Const*
_output_shapes
:*
dtype0*.
value%B#BhomeB
restaurantBoutdoor
h
Const_7Const*
_output_shapes
:*
dtype0	*-
value$B"	"                      
М
Const_8Const*
_output_shapes
:W*
dtype0	*–
value∆B√	W"Є                                                                	       
                                                                                                                                                                  !       "       #       $       %       &       '       (       )       *       +       ,       -       .       /       0       1       2       3       4       5       6       7       8       9       :       ;       <       =       >       ?       @       A       B       C       D       E       F       G       H       I       J       K       L       M       N       O       P       Q       R       S       T       U       V       
х	
Const_9Const*
_output_shapes
:W*
dtype0*є	
valueѓ	Bђ	WBlegumesBSoybeansBNotAllergensB	tree nutsBsoyBlactoseBdairyBpeanutsBgarlicBdairy, glutenB	shellfishBMilkBseafoodBglutenBeggsBlactose, eggsBsoy, peanutsBsoy, glutenBwheatBtree nuts, glutenBWheatBEggsBsesameB
soy, wheatBwheat, dairyBlactose, dairyBpoultryB	Tree nutsB	ShellfishB
nightshadeBtree nuts, dairyBseafood, lactoseBmeatBsesame, soy, peanuts, glutenBpeanuts, eggsBshellfish, peanutsBsoy, wheat, porkBwheat, tree nutsBFishBlegumes, dairyB	soy, porkBseafood, dairyBsoy, shellfishBsoy, shellfish, glutenBsoy, wheat, shellfishBlegumes, cornBporkBblack beansBseafood, citrusBdairy, eggsBeggs, glutenBpork, soy, glutenBsesame, soy, eggsBspicesBlegumes, glutenBseafood, soyBsoy, shellfish, eggsBpeanuts, glutenBshellfish, glutenBPeanutsBlactose, tree nutsBcilantroBcoconutBsesame, glutenBDairyBsoy, tree nutsBseafood, glutenBshellfish, dairyBsesame, soyBwheat, tree nuts, dairyBmustardBmustard, tree nutsBgluten, eggsBcitrusBgluten, soy, peanutsBsesame, eggsBcornBwheat, shellfishBseafood, shellfishBpork, glutenBwheat, eggsBshellfish, tree nutsBwheat, lactoseBCrustacean shellfishB	soy, eggsBriceBrice, gluten
ћ
Const_10Const*
_output_shapes
:*
dtype0	*П
valueЕBВ	"x                                                                	       
                                   
Д
Const_11Const*
_output_shapes
:*
dtype0*«
valueљBЇBveganB
meat-basedBkosherB
vegetarianBhalalBalcohol-basedBNotRestrictionBdessertBgrain-basedBpescatarianBseafood-basedBdairyBbeverageBketoBpescatarian, seafood-based
q
Const_12Const*
_output_shapes
:*
dtype0	*5
value,B*	"                              
z
Const_13Const*
_output_shapes
:*
dtype0*>
value5B3B
overweightBobesityBunderweightBhealthy
z
Const_14Const*
_output_shapes
:*
dtype0*>
value5B3B
overweightBunderweightBhealthyBobesity
q
Const_15Const*
_output_shapes
:*
dtype0	*5
value,B*	"                              
q
Const_16Const*
_output_shapes
:*
dtype0	*5
value,B*	"                              
l
Const_17Const*
_output_shapes
:*
dtype0*0
value'B%BBlackBLatinoBAsianBWhite
a
Const_18Const*
_output_shapes
:*
dtype0	*%
valueB	"               
`
Const_19Const*
_output_shapes
:*
dtype0*$
valueBBMarriedBSingle
q
Const_20Const*
_output_shapes
:*
dtype0	*5
value,B*	"                              
О
Const_21Const*
_output_shapes
:*
dtype0*R
valueIBGBFull-time-workerBHalf-time-workerBSelf-employeeB
Unemployed
Р
Const_22Const*
_output_shapes
:*
dtype0*T
valueKBIB
NotAllergyBwheatBpeanutB
cow's milkBsoyB	shellfishBMultiple
Й
Const_23Const*
_output_shapes
:*
dtype0	*M
valueDBB	"8                                                  
Б
Const_24Const*
_output_shapes
:*
dtype0	*E
value<B:	"0                                           
Ї
Const_25Const*
_output_shapes
:*
dtype0*~
valueuBsBNotRestrictionBflexi_observantBhalal_observantBkosher_observantBvegetarian_observantBvegan_observant
q
Const_26Const*
_output_shapes
:*
dtype0	*5
value,B*	"                              
К
Const_27Const*
_output_shapes
:*
dtype0*N
valueEBCBLightly activeBModerately activeBVery activeB	Sedentary
С
Const_28Const*
_output_shapes
:*
dtype0	*U
valueLBJ	"@                                                         
И
Const_29Const*
_output_shapes
:*
dtype0*L
valueCBAB18-29B70-79B90-100B60-69B40-49B50-59B80-89B30-39
U
Const_30Const*
_output_shapes
:*
dtype0*
valueBBFBM
a
Const_31Const*
_output_shapes
:*
dtype0	*%
valueB	"               
i
Const_32Const*
_output_shapes
:*
dtype0	*-
value$B"	"                      
w
Const_33Const*
_output_shapes
:*
dtype0*;
value2B0Blose_weightBgain_weightBmaintain_fit
S
Const_34Const*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€
S
Const_35Const*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€
S
Const_36Const*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€
S
Const_37Const*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€
S
Const_38Const*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€
S
Const_39Const*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€
S
Const_40Const*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€
S
Const_41Const*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€
S
Const_42Const*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€
S
Const_43Const*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€
S
Const_44Const*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€
S
Const_45Const*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€
S
Const_46Const*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€
S
Const_47Const*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€
S
Const_48Const*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€
S
Const_49Const*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€
S
Const_50Const*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€
Ш
false_negativesVarHandleOp*
_output_shapes
: * 

debug_namefalse_negatives/*
dtype0*
shape:* 
shared_namefalse_negatives
o
#false_negatives/Read/ReadVariableOpReadVariableOpfalse_negatives*
_output_shapes
:*
dtype0
Х
true_positivesVarHandleOp*
_output_shapes
: *

debug_nametrue_positives/*
dtype0*
shape:*
shared_nametrue_positives
m
"true_positives/Read/ReadVariableOpReadVariableOptrue_positives*
_output_shapes
:*
dtype0
Ш
false_positivesVarHandleOp*
_output_shapes
: * 

debug_namefalse_positives/*
dtype0*
shape:* 
shared_namefalse_positives
o
#false_positives/Read/ReadVariableOpReadVariableOpfalse_positives*
_output_shapes
:*
dtype0
Ы
true_positives_1VarHandleOp*
_output_shapes
: *!

debug_nametrue_positives_1/*
dtype0*
shape:*!
shared_nametrue_positives_1
q
$true_positives_1/Read/ReadVariableOpReadVariableOptrue_positives_1*
_output_shapes
:*
dtype0
v
countVarHandleOp*
_output_shapes
: *

debug_namecount/*
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
v
totalVarHandleOp*
_output_shapes
: *

debug_nametotal/*
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
|
count_1VarHandleOp*
_output_shapes
: *

debug_name
count_1/*
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
|
total_1VarHandleOp*
_output_shapes
: *

debug_name
total_1/*
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
І
Adam/v/output_0/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/v/output_0/bias/*
dtype0*
shape:*%
shared_nameAdam/v/output_0/bias
y
(Adam/v/output_0/bias/Read/ReadVariableOpReadVariableOpAdam/v/output_0/bias*
_output_shapes
:*
dtype0
І
Adam/m/output_0/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/m/output_0/bias/*
dtype0*
shape:*%
shared_nameAdam/m/output_0/bias
y
(Adam/m/output_0/bias/Read/ReadVariableOpReadVariableOpAdam/m/output_0/bias*
_output_shapes
:*
dtype0
±
Adam/v/output_0/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/v/output_0/kernel/*
dtype0*
shape
: *'
shared_nameAdam/v/output_0/kernel
Б
*Adam/v/output_0/kernel/Read/ReadVariableOpReadVariableOpAdam/v/output_0/kernel*
_output_shapes

: *
dtype0
±
Adam/m/output_0/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/m/output_0/kernel/*
dtype0*
shape
: *'
shared_nameAdam/m/output_0/kernel
Б
*Adam/m/output_0/kernel/Read/ReadVariableOpReadVariableOpAdam/m/output_0/kernel*
_output_shapes

: *
dtype0
≠
Adam/v/fc_layer_2/biasVarHandleOp*
_output_shapes
: *'

debug_nameAdam/v/fc_layer_2/bias/*
dtype0*
shape: *'
shared_nameAdam/v/fc_layer_2/bias
}
*Adam/v/fc_layer_2/bias/Read/ReadVariableOpReadVariableOpAdam/v/fc_layer_2/bias*
_output_shapes
: *
dtype0
≠
Adam/m/fc_layer_2/biasVarHandleOp*
_output_shapes
: *'

debug_nameAdam/m/fc_layer_2/bias/*
dtype0*
shape: *'
shared_nameAdam/m/fc_layer_2/bias
}
*Adam/m/fc_layer_2/bias/Read/ReadVariableOpReadVariableOpAdam/m/fc_layer_2/bias*
_output_shapes
: *
dtype0
Ј
Adam/v/fc_layer_2/kernelVarHandleOp*
_output_shapes
: *)

debug_nameAdam/v/fc_layer_2/kernel/*
dtype0*
shape
:@ *)
shared_nameAdam/v/fc_layer_2/kernel
Е
,Adam/v/fc_layer_2/kernel/Read/ReadVariableOpReadVariableOpAdam/v/fc_layer_2/kernel*
_output_shapes

:@ *
dtype0
Ј
Adam/m/fc_layer_2/kernelVarHandleOp*
_output_shapes
: *)

debug_nameAdam/m/fc_layer_2/kernel/*
dtype0*
shape
:@ *)
shared_nameAdam/m/fc_layer_2/kernel
Е
,Adam/m/fc_layer_2/kernel/Read/ReadVariableOpReadVariableOpAdam/m/fc_layer_2/kernel*
_output_shapes

:@ *
dtype0
≠
Adam/v/fc_layer_1/biasVarHandleOp*
_output_shapes
: *'

debug_nameAdam/v/fc_layer_1/bias/*
dtype0*
shape:@*'
shared_nameAdam/v/fc_layer_1/bias
}
*Adam/v/fc_layer_1/bias/Read/ReadVariableOpReadVariableOpAdam/v/fc_layer_1/bias*
_output_shapes
:@*
dtype0
≠
Adam/m/fc_layer_1/biasVarHandleOp*
_output_shapes
: *'

debug_nameAdam/m/fc_layer_1/bias/*
dtype0*
shape:@*'
shared_nameAdam/m/fc_layer_1/bias
}
*Adam/m/fc_layer_1/bias/Read/ReadVariableOpReadVariableOpAdam/m/fc_layer_1/bias*
_output_shapes
:@*
dtype0
Є
Adam/v/fc_layer_1/kernelVarHandleOp*
_output_shapes
: *)

debug_nameAdam/v/fc_layer_1/kernel/*
dtype0*
shape:	А@*)
shared_nameAdam/v/fc_layer_1/kernel
Ж
,Adam/v/fc_layer_1/kernel/Read/ReadVariableOpReadVariableOpAdam/v/fc_layer_1/kernel*
_output_shapes
:	А@*
dtype0
Є
Adam/m/fc_layer_1/kernelVarHandleOp*
_output_shapes
: *)

debug_nameAdam/m/fc_layer_1/kernel/*
dtype0*
shape:	А@*)
shared_nameAdam/m/fc_layer_1/kernel
Ж
,Adam/m/fc_layer_1/kernel/Read/ReadVariableOpReadVariableOpAdam/m/fc_layer_1/kernel*
_output_shapes
:	А@*
dtype0
Ѓ
Adam/v/fc_layer_0/biasVarHandleOp*
_output_shapes
: *'

debug_nameAdam/v/fc_layer_0/bias/*
dtype0*
shape:А*'
shared_nameAdam/v/fc_layer_0/bias
~
*Adam/v/fc_layer_0/bias/Read/ReadVariableOpReadVariableOpAdam/v/fc_layer_0/bias*
_output_shapes	
:А*
dtype0
Ѓ
Adam/m/fc_layer_0/biasVarHandleOp*
_output_shapes
: *'

debug_nameAdam/m/fc_layer_0/bias/*
dtype0*
shape:А*'
shared_nameAdam/m/fc_layer_0/bias
~
*Adam/m/fc_layer_0/bias/Read/ReadVariableOpReadVariableOpAdam/m/fc_layer_0/bias*
_output_shapes	
:А*
dtype0
є
Adam/v/fc_layer_0/kernelVarHandleOp*
_output_shapes
: *)

debug_nameAdam/v/fc_layer_0/kernel/*
dtype0*
shape:
йА*)
shared_nameAdam/v/fc_layer_0/kernel
З
,Adam/v/fc_layer_0/kernel/Read/ReadVariableOpReadVariableOpAdam/v/fc_layer_0/kernel* 
_output_shapes
:
йА*
dtype0
є
Adam/m/fc_layer_0/kernelVarHandleOp*
_output_shapes
: *)

debug_nameAdam/m/fc_layer_0/kernel/*
dtype0*
shape:
йА*)
shared_nameAdam/m/fc_layer_0/kernel
З
,Adam/m/fc_layer_0/kernel/Read/ReadVariableOpReadVariableOpAdam/m/fc_layer_0/kernel* 
_output_shapes
:
йА*
dtype0
“
"Adam/v/batch_normalization_12/betaVarHandleOp*
_output_shapes
: *3

debug_name%#Adam/v/batch_normalization_12/beta/*
dtype0*
shape:й*3
shared_name$"Adam/v/batch_normalization_12/beta
Ц
6Adam/v/batch_normalization_12/beta/Read/ReadVariableOpReadVariableOp"Adam/v/batch_normalization_12/beta*
_output_shapes	
:й*
dtype0
“
"Adam/m/batch_normalization_12/betaVarHandleOp*
_output_shapes
: *3

debug_name%#Adam/m/batch_normalization_12/beta/*
dtype0*
shape:й*3
shared_name$"Adam/m/batch_normalization_12/beta
Ц
6Adam/m/batch_normalization_12/beta/Read/ReadVariableOpReadVariableOp"Adam/m/batch_normalization_12/beta*
_output_shapes	
:й*
dtype0
’
#Adam/v/batch_normalization_12/gammaVarHandleOp*
_output_shapes
: *4

debug_name&$Adam/v/batch_normalization_12/gamma/*
dtype0*
shape:й*4
shared_name%#Adam/v/batch_normalization_12/gamma
Ш
7Adam/v/batch_normalization_12/gamma/Read/ReadVariableOpReadVariableOp#Adam/v/batch_normalization_12/gamma*
_output_shapes	
:й*
dtype0
’
#Adam/m/batch_normalization_12/gammaVarHandleOp*
_output_shapes
: *4

debug_name&$Adam/m/batch_normalization_12/gamma/*
dtype0*
shape:й*4
shared_name%#Adam/m/batch_normalization_12/gamma
Ш
7Adam/m/batch_normalization_12/gamma/Read/ReadVariableOpReadVariableOp#Adam/m/batch_normalization_12/gamma*
_output_shapes	
:й*
dtype0
¬
Adam/v/context_embedding/biasVarHandleOp*
_output_shapes
: *.

debug_name Adam/v/context_embedding/bias/*
dtype0*
shape:*.
shared_nameAdam/v/context_embedding/bias
Л
1Adam/v/context_embedding/bias/Read/ReadVariableOpReadVariableOpAdam/v/context_embedding/bias*
_output_shapes
:*
dtype0
¬
Adam/m/context_embedding/biasVarHandleOp*
_output_shapes
: *.

debug_name Adam/m/context_embedding/bias/*
dtype0*
shape:*.
shared_nameAdam/m/context_embedding/bias
Л
1Adam/m/context_embedding/bias/Read/ReadVariableOpReadVariableOpAdam/m/context_embedding/bias*
_output_shapes
:*
dtype0
Ќ
Adam/v/context_embedding/kernelVarHandleOp*
_output_shapes
: *0

debug_name" Adam/v/context_embedding/kernel/*
dtype0*
shape:	Ш*0
shared_name!Adam/v/context_embedding/kernel
Ф
3Adam/v/context_embedding/kernel/Read/ReadVariableOpReadVariableOpAdam/v/context_embedding/kernel*
_output_shapes
:	Ш*
dtype0
Ќ
Adam/m/context_embedding/kernelVarHandleOp*
_output_shapes
: *0

debug_name" Adam/m/context_embedding/kernel/*
dtype0*
shape:	Ш*0
shared_name!Adam/m/context_embedding/kernel
Ф
3Adam/m/context_embedding/kernel/Read/ReadVariableOpReadVariableOpAdam/m/context_embedding/kernel*
_output_shapes
:	Ш*
dtype0
Ї
Adam/v/food_embedding/biasVarHandleOp*
_output_shapes
: *+

debug_nameAdam/v/food_embedding/bias/*
dtype0*
shape:ђ*+
shared_nameAdam/v/food_embedding/bias
Ж
.Adam/v/food_embedding/bias/Read/ReadVariableOpReadVariableOpAdam/v/food_embedding/bias*
_output_shapes	
:ђ*
dtype0
Ї
Adam/m/food_embedding/biasVarHandleOp*
_output_shapes
: *+

debug_nameAdam/m/food_embedding/bias/*
dtype0*
shape:ђ*+
shared_nameAdam/m/food_embedding/bias
Ж
.Adam/m/food_embedding/bias/Read/ReadVariableOpReadVariableOpAdam/m/food_embedding/bias*
_output_shapes	
:ђ*
dtype0
≈
Adam/v/food_embedding/kernelVarHandleOp*
_output_shapes
: *-

debug_nameAdam/v/food_embedding/kernel/*
dtype0*
shape:
£ђ*-
shared_nameAdam/v/food_embedding/kernel
П
0Adam/v/food_embedding/kernel/Read/ReadVariableOpReadVariableOpAdam/v/food_embedding/kernel* 
_output_shapes
:
£ђ*
dtype0
≈
Adam/m/food_embedding/kernelVarHandleOp*
_output_shapes
: *-

debug_nameAdam/m/food_embedding/kernel/*
dtype0*
shape:
£ђ*-
shared_nameAdam/m/food_embedding/kernel
П
0Adam/m/food_embedding/kernel/Read/ReadVariableOpReadVariableOpAdam/m/food_embedding/kernel* 
_output_shapes
:
£ђ*
dtype0
Ї
Adam/v/user_embedding/biasVarHandleOp*
_output_shapes
: *+

debug_nameAdam/v/user_embedding/bias/*
dtype0*
shape:ђ*+
shared_nameAdam/v/user_embedding/bias
Ж
.Adam/v/user_embedding/bias/Read/ReadVariableOpReadVariableOpAdam/v/user_embedding/bias*
_output_shapes	
:ђ*
dtype0
Ї
Adam/m/user_embedding/biasVarHandleOp*
_output_shapes
: *+

debug_nameAdam/m/user_embedding/bias/*
dtype0*
shape:ђ*+
shared_nameAdam/m/user_embedding/bias
Ж
.Adam/m/user_embedding/bias/Read/ReadVariableOpReadVariableOpAdam/m/user_embedding/bias*
_output_shapes	
:ђ*
dtype0
ƒ
Adam/v/user_embedding/kernelVarHandleOp*
_output_shapes
: *-

debug_nameAdam/v/user_embedding/kernel/*
dtype0*
shape:	ђ*-
shared_nameAdam/v/user_embedding/kernel
О
0Adam/v/user_embedding/kernel/Read/ReadVariableOpReadVariableOpAdam/v/user_embedding/kernel*
_output_shapes
:	ђ*
dtype0
ƒ
Adam/m/user_embedding/kernelVarHandleOp*
_output_shapes
: *-

debug_nameAdam/m/user_embedding/kernel/*
dtype0*
shape:	ђ*-
shared_nameAdam/m/user_embedding/kernel
О
0Adam/m/user_embedding/kernel/Read/ReadVariableOpReadVariableOpAdam/m/user_embedding/kernel*
_output_shapes
:	ђ*
dtype0
…
Adam/v/embedding_88/embeddingsVarHandleOp*
_output_shapes
: */

debug_name!Adam/v/embedding_88/embeddings/*
dtype0*
shape
:*/
shared_name Adam/v/embedding_88/embeddings
С
2Adam/v/embedding_88/embeddings/Read/ReadVariableOpReadVariableOpAdam/v/embedding_88/embeddings*
_output_shapes

:*
dtype0
…
Adam/m/embedding_88/embeddingsVarHandleOp*
_output_shapes
: */

debug_name!Adam/m/embedding_88/embeddings/*
dtype0*
shape
:*/
shared_name Adam/m/embedding_88/embeddings
С
2Adam/m/embedding_88/embeddings/Read/ReadVariableOpReadVariableOpAdam/m/embedding_88/embeddings*
_output_shapes

:*
dtype0
…
Adam/v/embedding_87/embeddingsVarHandleOp*
_output_shapes
: */

debug_name!Adam/v/embedding_87/embeddings/*
dtype0*
shape
:*/
shared_name Adam/v/embedding_87/embeddings
С
2Adam/v/embedding_87/embeddings/Read/ReadVariableOpReadVariableOpAdam/v/embedding_87/embeddings*
_output_shapes

:*
dtype0
…
Adam/m/embedding_87/embeddingsVarHandleOp*
_output_shapes
: */

debug_name!Adam/m/embedding_87/embeddings/*
dtype0*
shape
:*/
shared_name Adam/m/embedding_87/embeddings
С
2Adam/m/embedding_87/embeddings/Read/ReadVariableOpReadVariableOpAdam/m/embedding_87/embeddings*
_output_shapes

:*
dtype0
…
Adam/v/embedding_90/embeddingsVarHandleOp*
_output_shapes
: */

debug_name!Adam/v/embedding_90/embeddings/*
dtype0*
shape
:W	*/
shared_name Adam/v/embedding_90/embeddings
С
2Adam/v/embedding_90/embeddings/Read/ReadVariableOpReadVariableOpAdam/v/embedding_90/embeddings*
_output_shapes

:W	*
dtype0
…
Adam/m/embedding_90/embeddingsVarHandleOp*
_output_shapes
: */

debug_name!Adam/m/embedding_90/embeddings/*
dtype0*
shape
:W	*/
shared_name Adam/m/embedding_90/embeddings
С
2Adam/m/embedding_90/embeddings/Read/ReadVariableOpReadVariableOpAdam/m/embedding_90/embeddings*
_output_shapes

:W	*
dtype0
…
Adam/v/embedding_89/embeddingsVarHandleOp*
_output_shapes
: */

debug_name!Adam/v/embedding_89/embeddings/*
dtype0*
shape
:*/
shared_name Adam/v/embedding_89/embeddings
С
2Adam/v/embedding_89/embeddings/Read/ReadVariableOpReadVariableOpAdam/v/embedding_89/embeddings*
_output_shapes

:*
dtype0
…
Adam/m/embedding_89/embeddingsVarHandleOp*
_output_shapes
: */

debug_name!Adam/m/embedding_89/embeddings/*
dtype0*
shape
:*/
shared_name Adam/m/embedding_89/embeddings
С
2Adam/m/embedding_89/embeddings/Read/ReadVariableOpReadVariableOpAdam/m/embedding_89/embeddings*
_output_shapes

:*
dtype0
…
Adam/v/embedding_86/embeddingsVarHandleOp*
_output_shapes
: */

debug_name!Adam/v/embedding_86/embeddings/*
dtype0*
shape
:*/
shared_name Adam/v/embedding_86/embeddings
С
2Adam/v/embedding_86/embeddings/Read/ReadVariableOpReadVariableOpAdam/v/embedding_86/embeddings*
_output_shapes

:*
dtype0
…
Adam/m/embedding_86/embeddingsVarHandleOp*
_output_shapes
: */

debug_name!Adam/m/embedding_86/embeddings/*
dtype0*
shape
:*/
shared_name Adam/m/embedding_86/embeddings
С
2Adam/m/embedding_86/embeddings/Read/ReadVariableOpReadVariableOpAdam/m/embedding_86/embeddings*
_output_shapes

:*
dtype0
…
Adam/v/embedding_85/embeddingsVarHandleOp*
_output_shapes
: */

debug_name!Adam/v/embedding_85/embeddings/*
dtype0*
shape
:*/
shared_name Adam/v/embedding_85/embeddings
С
2Adam/v/embedding_85/embeddings/Read/ReadVariableOpReadVariableOpAdam/v/embedding_85/embeddings*
_output_shapes

:*
dtype0
…
Adam/m/embedding_85/embeddingsVarHandleOp*
_output_shapes
: */

debug_name!Adam/m/embedding_85/embeddings/*
dtype0*
shape
:*/
shared_name Adam/m/embedding_85/embeddings
С
2Adam/m/embedding_85/embeddings/Read/ReadVariableOpReadVariableOpAdam/m/embedding_85/embeddings*
_output_shapes

:*
dtype0
…
Adam/v/embedding_84/embeddingsVarHandleOp*
_output_shapes
: */

debug_name!Adam/v/embedding_84/embeddings/*
dtype0*
shape
:*/
shared_name Adam/v/embedding_84/embeddings
С
2Adam/v/embedding_84/embeddings/Read/ReadVariableOpReadVariableOpAdam/v/embedding_84/embeddings*
_output_shapes

:*
dtype0
…
Adam/m/embedding_84/embeddingsVarHandleOp*
_output_shapes
: */

debug_name!Adam/m/embedding_84/embeddings/*
dtype0*
shape
:*/
shared_name Adam/m/embedding_84/embeddings
С
2Adam/m/embedding_84/embeddings/Read/ReadVariableOpReadVariableOpAdam/m/embedding_84/embeddings*
_output_shapes

:*
dtype0
…
Adam/v/embedding_83/embeddingsVarHandleOp*
_output_shapes
: */

debug_name!Adam/v/embedding_83/embeddings/*
dtype0*
shape
:*/
shared_name Adam/v/embedding_83/embeddings
С
2Adam/v/embedding_83/embeddings/Read/ReadVariableOpReadVariableOpAdam/v/embedding_83/embeddings*
_output_shapes

:*
dtype0
…
Adam/m/embedding_83/embeddingsVarHandleOp*
_output_shapes
: */

debug_name!Adam/m/embedding_83/embeddings/*
dtype0*
shape
:*/
shared_name Adam/m/embedding_83/embeddings
С
2Adam/m/embedding_83/embeddings/Read/ReadVariableOpReadVariableOpAdam/m/embedding_83/embeddings*
_output_shapes

:*
dtype0
…
Adam/v/embedding_82/embeddingsVarHandleOp*
_output_shapes
: */

debug_name!Adam/v/embedding_82/embeddings/*
dtype0*
shape
:*/
shared_name Adam/v/embedding_82/embeddings
С
2Adam/v/embedding_82/embeddings/Read/ReadVariableOpReadVariableOpAdam/v/embedding_82/embeddings*
_output_shapes

:*
dtype0
…
Adam/m/embedding_82/embeddingsVarHandleOp*
_output_shapes
: */

debug_name!Adam/m/embedding_82/embeddings/*
dtype0*
shape
:*/
shared_name Adam/m/embedding_82/embeddings
С
2Adam/m/embedding_82/embeddings/Read/ReadVariableOpReadVariableOpAdam/m/embedding_82/embeddings*
_output_shapes

:*
dtype0
…
Adam/v/embedding_81/embeddingsVarHandleOp*
_output_shapes
: */

debug_name!Adam/v/embedding_81/embeddings/*
dtype0*
shape
:*/
shared_name Adam/v/embedding_81/embeddings
С
2Adam/v/embedding_81/embeddings/Read/ReadVariableOpReadVariableOpAdam/v/embedding_81/embeddings*
_output_shapes

:*
dtype0
…
Adam/m/embedding_81/embeddingsVarHandleOp*
_output_shapes
: */

debug_name!Adam/m/embedding_81/embeddings/*
dtype0*
shape
:*/
shared_name Adam/m/embedding_81/embeddings
С
2Adam/m/embedding_81/embeddings/Read/ReadVariableOpReadVariableOpAdam/m/embedding_81/embeddings*
_output_shapes

:*
dtype0
…
Adam/v/embedding_80/embeddingsVarHandleOp*
_output_shapes
: */

debug_name!Adam/v/embedding_80/embeddings/*
dtype0*
shape
:*/
shared_name Adam/v/embedding_80/embeddings
С
2Adam/v/embedding_80/embeddings/Read/ReadVariableOpReadVariableOpAdam/v/embedding_80/embeddings*
_output_shapes

:*
dtype0
…
Adam/m/embedding_80/embeddingsVarHandleOp*
_output_shapes
: */

debug_name!Adam/m/embedding_80/embeddings/*
dtype0*
shape
:*/
shared_name Adam/m/embedding_80/embeddings
С
2Adam/m/embedding_80/embeddings/Read/ReadVariableOpReadVariableOpAdam/m/embedding_80/embeddings*
_output_shapes

:*
dtype0
…
Adam/v/embedding_79/embeddingsVarHandleOp*
_output_shapes
: */

debug_name!Adam/v/embedding_79/embeddings/*
dtype0*
shape
:*/
shared_name Adam/v/embedding_79/embeddings
С
2Adam/v/embedding_79/embeddings/Read/ReadVariableOpReadVariableOpAdam/v/embedding_79/embeddings*
_output_shapes

:*
dtype0
…
Adam/m/embedding_79/embeddingsVarHandleOp*
_output_shapes
: */

debug_name!Adam/m/embedding_79/embeddings/*
dtype0*
shape
:*/
shared_name Adam/m/embedding_79/embeddings
С
2Adam/m/embedding_79/embeddings/Read/ReadVariableOpReadVariableOpAdam/m/embedding_79/embeddings*
_output_shapes

:*
dtype0
…
Adam/v/embedding_78/embeddingsVarHandleOp*
_output_shapes
: */

debug_name!Adam/v/embedding_78/embeddings/*
dtype0*
shape
:*/
shared_name Adam/v/embedding_78/embeddings
С
2Adam/v/embedding_78/embeddings/Read/ReadVariableOpReadVariableOpAdam/v/embedding_78/embeddings*
_output_shapes

:*
dtype0
…
Adam/m/embedding_78/embeddingsVarHandleOp*
_output_shapes
: */

debug_name!Adam/m/embedding_78/embeddings/*
dtype0*
shape
:*/
shared_name Adam/m/embedding_78/embeddings
С
2Adam/m/embedding_78/embeddings/Read/ReadVariableOpReadVariableOpAdam/m/embedding_78/embeddings*
_output_shapes

:*
dtype0
…
Adam/v/embedding_77/embeddingsVarHandleOp*
_output_shapes
: */

debug_name!Adam/v/embedding_77/embeddings/*
dtype0*
shape
:*/
shared_name Adam/v/embedding_77/embeddings
С
2Adam/v/embedding_77/embeddings/Read/ReadVariableOpReadVariableOpAdam/v/embedding_77/embeddings*
_output_shapes

:*
dtype0
…
Adam/m/embedding_77/embeddingsVarHandleOp*
_output_shapes
: */

debug_name!Adam/m/embedding_77/embeddings/*
dtype0*
shape
:*/
shared_name Adam/m/embedding_77/embeddings
С
2Adam/m/embedding_77/embeddings/Read/ReadVariableOpReadVariableOpAdam/m/embedding_77/embeddings*
_output_shapes

:*
dtype0
…
Adam/v/embedding_76/embeddingsVarHandleOp*
_output_shapes
: */

debug_name!Adam/v/embedding_76/embeddings/*
dtype0*
shape
:*/
shared_name Adam/v/embedding_76/embeddings
С
2Adam/v/embedding_76/embeddings/Read/ReadVariableOpReadVariableOpAdam/v/embedding_76/embeddings*
_output_shapes

:*
dtype0
…
Adam/m/embedding_76/embeddingsVarHandleOp*
_output_shapes
: */

debug_name!Adam/m/embedding_76/embeddings/*
dtype0*
shape
:*/
shared_name Adam/m/embedding_76/embeddings
С
2Adam/m/embedding_76/embeddings/Read/ReadVariableOpReadVariableOpAdam/m/embedding_76/embeddings*
_output_shapes

:*
dtype0
О
learning_rateVarHandleOp*
_output_shapes
: *

debug_namelearning_rate/*
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
В
	iterationVarHandleOp*
_output_shapes
: *

debug_name
iteration/*
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
Т
output_0/biasVarHandleOp*
_output_shapes
: *

debug_nameoutput_0/bias/*
dtype0*
shape:*
shared_nameoutput_0/bias
k
!output_0/bias/Read/ReadVariableOpReadVariableOpoutput_0/bias*
_output_shapes
:*
dtype0
Ь
output_0/kernelVarHandleOp*
_output_shapes
: * 

debug_nameoutput_0/kernel/*
dtype0*
shape
: * 
shared_nameoutput_0/kernel
s
#output_0/kernel/Read/ReadVariableOpReadVariableOpoutput_0/kernel*
_output_shapes

: *
dtype0
Ш
fc_layer_2/biasVarHandleOp*
_output_shapes
: * 

debug_namefc_layer_2/bias/*
dtype0*
shape: * 
shared_namefc_layer_2/bias
o
#fc_layer_2/bias/Read/ReadVariableOpReadVariableOpfc_layer_2/bias*
_output_shapes
: *
dtype0
Ґ
fc_layer_2/kernelVarHandleOp*
_output_shapes
: *"

debug_namefc_layer_2/kernel/*
dtype0*
shape
:@ *"
shared_namefc_layer_2/kernel
w
%fc_layer_2/kernel/Read/ReadVariableOpReadVariableOpfc_layer_2/kernel*
_output_shapes

:@ *
dtype0
Ш
fc_layer_1/biasVarHandleOp*
_output_shapes
: * 

debug_namefc_layer_1/bias/*
dtype0*
shape:@* 
shared_namefc_layer_1/bias
o
#fc_layer_1/bias/Read/ReadVariableOpReadVariableOpfc_layer_1/bias*
_output_shapes
:@*
dtype0
£
fc_layer_1/kernelVarHandleOp*
_output_shapes
: *"

debug_namefc_layer_1/kernel/*
dtype0*
shape:	А@*"
shared_namefc_layer_1/kernel
x
%fc_layer_1/kernel/Read/ReadVariableOpReadVariableOpfc_layer_1/kernel*
_output_shapes
:	А@*
dtype0
Щ
fc_layer_0/biasVarHandleOp*
_output_shapes
: * 

debug_namefc_layer_0/bias/*
dtype0*
shape:А* 
shared_namefc_layer_0/bias
p
#fc_layer_0/bias/Read/ReadVariableOpReadVariableOpfc_layer_0/bias*
_output_shapes	
:А*
dtype0
§
fc_layer_0/kernelVarHandleOp*
_output_shapes
: *"

debug_namefc_layer_0/kernel/*
dtype0*
shape:
йА*"
shared_namefc_layer_0/kernel
y
%fc_layer_0/kernel/Read/ReadVariableOpReadVariableOpfc_layer_0/kernel* 
_output_shapes
:
йА*
dtype0
ё
&batch_normalization_12/moving_varianceVarHandleOp*
_output_shapes
: *7

debug_name)'batch_normalization_12/moving_variance/*
dtype0*
shape:й*7
shared_name(&batch_normalization_12/moving_variance
Ю
:batch_normalization_12/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_12/moving_variance*
_output_shapes	
:й*
dtype0
“
"batch_normalization_12/moving_meanVarHandleOp*
_output_shapes
: *3

debug_name%#batch_normalization_12/moving_mean/*
dtype0*
shape:й*3
shared_name$"batch_normalization_12/moving_mean
Ц
6batch_normalization_12/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_12/moving_mean*
_output_shapes	
:й*
dtype0
љ
batch_normalization_12/betaVarHandleOp*
_output_shapes
: *,

debug_namebatch_normalization_12/beta/*
dtype0*
shape:й*,
shared_namebatch_normalization_12/beta
И
/batch_normalization_12/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_12/beta*
_output_shapes	
:й*
dtype0
ј
batch_normalization_12/gammaVarHandleOp*
_output_shapes
: *-

debug_namebatch_normalization_12/gamma/*
dtype0*
shape:й*-
shared_namebatch_normalization_12/gamma
К
0batch_normalization_12/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_12/gamma*
_output_shapes	
:й*
dtype0
≠
context_embedding/biasVarHandleOp*
_output_shapes
: *'

debug_namecontext_embedding/bias/*
dtype0*
shape:*'
shared_namecontext_embedding/bias
}
*context_embedding/bias/Read/ReadVariableOpReadVariableOpcontext_embedding/bias*
_output_shapes
:*
dtype0
Є
context_embedding/kernelVarHandleOp*
_output_shapes
: *)

debug_namecontext_embedding/kernel/*
dtype0*
shape:	Ш*)
shared_namecontext_embedding/kernel
Ж
,context_embedding/kernel/Read/ReadVariableOpReadVariableOpcontext_embedding/kernel*
_output_shapes
:	Ш*
dtype0
•
food_embedding/biasVarHandleOp*
_output_shapes
: *$

debug_namefood_embedding/bias/*
dtype0*
shape:ђ*$
shared_namefood_embedding/bias
x
'food_embedding/bias/Read/ReadVariableOpReadVariableOpfood_embedding/bias*
_output_shapes	
:ђ*
dtype0
∞
food_embedding/kernelVarHandleOp*
_output_shapes
: *&

debug_namefood_embedding/kernel/*
dtype0*
shape:
£ђ*&
shared_namefood_embedding/kernel
Б
)food_embedding/kernel/Read/ReadVariableOpReadVariableOpfood_embedding/kernel* 
_output_shapes
:
£ђ*
dtype0
•
user_embedding/biasVarHandleOp*
_output_shapes
: *$

debug_nameuser_embedding/bias/*
dtype0*
shape:ђ*$
shared_nameuser_embedding/bias
x
'user_embedding/bias/Read/ReadVariableOpReadVariableOpuser_embedding/bias*
_output_shapes	
:ђ*
dtype0
ѓ
user_embedding/kernelVarHandleOp*
_output_shapes
: *&

debug_nameuser_embedding/kernel/*
dtype0*
shape:	ђ*&
shared_nameuser_embedding/kernel
А
)user_embedding/kernel/Read/ReadVariableOpReadVariableOpuser_embedding/kernel*
_output_shapes
:	ђ*
dtype0
o

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	6434959*
value_dtype0	
і
embedding_88/embeddingsVarHandleOp*
_output_shapes
: *(

debug_nameembedding_88/embeddings/*
dtype0*
shape
:*(
shared_nameembedding_88/embeddings
Г
+embedding_88/embeddings/Read/ReadVariableOpReadVariableOpembedding_88/embeddings*
_output_shapes

:*
dtype0
і
embedding_87/embeddingsVarHandleOp*
_output_shapes
: *(

debug_nameembedding_87/embeddings/*
dtype0*
shape
:*(
shared_nameembedding_87/embeddings
Г
+embedding_87/embeddings/Read/ReadVariableOpReadVariableOpembedding_87/embeddings*
_output_shapes

:*
dtype0
q
hash_table_1HashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	6435241*
value_dtype0	
q
hash_table_2HashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	6435054*
value_dtype0	
q
hash_table_3HashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	6435003*
value_dtype0	
і
embedding_90/embeddingsVarHandleOp*
_output_shapes
: *(

debug_nameembedding_90/embeddings/*
dtype0*
shape
:W	*(
shared_nameembedding_90/embeddings
Г
+embedding_90/embeddings/Read/ReadVariableOpReadVariableOpembedding_90/embeddings*
_output_shapes

:W	*
dtype0
і
embedding_89/embeddingsVarHandleOp*
_output_shapes
: *(

debug_nameembedding_89/embeddings/*
dtype0*
shape
:*(
shared_nameembedding_89/embeddings
Г
+embedding_89/embeddings/Read/ReadVariableOpReadVariableOpembedding_89/embeddings*
_output_shapes

:*
dtype0
і
embedding_86/embeddingsVarHandleOp*
_output_shapes
: *(

debug_nameembedding_86/embeddings/*
dtype0*
shape
:*(
shared_nameembedding_86/embeddings
Г
+embedding_86/embeddings/Read/ReadVariableOpReadVariableOpembedding_86/embeddings*
_output_shapes

:*
dtype0
і
embedding_85/embeddingsVarHandleOp*
_output_shapes
: *(

debug_nameembedding_85/embeddings/*
dtype0*
shape
:*(
shared_nameembedding_85/embeddings
Г
+embedding_85/embeddings/Read/ReadVariableOpReadVariableOpembedding_85/embeddings*
_output_shapes

:*
dtype0
і
embedding_84/embeddingsVarHandleOp*
_output_shapes
: *(

debug_nameembedding_84/embeddings/*
dtype0*
shape
:*(
shared_nameembedding_84/embeddings
Г
+embedding_84/embeddings/Read/ReadVariableOpReadVariableOpembedding_84/embeddings*
_output_shapes

:*
dtype0
і
embedding_83/embeddingsVarHandleOp*
_output_shapes
: *(

debug_nameembedding_83/embeddings/*
dtype0*
shape
:*(
shared_nameembedding_83/embeddings
Г
+embedding_83/embeddings/Read/ReadVariableOpReadVariableOpembedding_83/embeddings*
_output_shapes

:*
dtype0
і
embedding_82/embeddingsVarHandleOp*
_output_shapes
: *(

debug_nameembedding_82/embeddings/*
dtype0*
shape
:*(
shared_nameembedding_82/embeddings
Г
+embedding_82/embeddings/Read/ReadVariableOpReadVariableOpembedding_82/embeddings*
_output_shapes

:*
dtype0
і
embedding_81/embeddingsVarHandleOp*
_output_shapes
: *(

debug_nameembedding_81/embeddings/*
dtype0*
shape
:*(
shared_nameembedding_81/embeddings
Г
+embedding_81/embeddings/Read/ReadVariableOpReadVariableOpembedding_81/embeddings*
_output_shapes

:*
dtype0
і
embedding_80/embeddingsVarHandleOp*
_output_shapes
: *(

debug_nameembedding_80/embeddings/*
dtype0*
shape
:*(
shared_nameembedding_80/embeddings
Г
+embedding_80/embeddings/Read/ReadVariableOpReadVariableOpembedding_80/embeddings*
_output_shapes

:*
dtype0
і
embedding_79/embeddingsVarHandleOp*
_output_shapes
: *(

debug_nameembedding_79/embeddings/*
dtype0*
shape
:*(
shared_nameembedding_79/embeddings
Г
+embedding_79/embeddings/Read/ReadVariableOpReadVariableOpembedding_79/embeddings*
_output_shapes

:*
dtype0
і
embedding_78/embeddingsVarHandleOp*
_output_shapes
: *(

debug_nameembedding_78/embeddings/*
dtype0*
shape
:*(
shared_nameembedding_78/embeddings
Г
+embedding_78/embeddings/Read/ReadVariableOpReadVariableOpembedding_78/embeddings*
_output_shapes

:*
dtype0
і
embedding_77/embeddingsVarHandleOp*
_output_shapes
: *(

debug_nameembedding_77/embeddings/*
dtype0*
shape
:*(
shared_nameembedding_77/embeddings
Г
+embedding_77/embeddings/Read/ReadVariableOpReadVariableOpembedding_77/embeddings*
_output_shapes

:*
dtype0
і
embedding_76/embeddingsVarHandleOp*
_output_shapes
: *(

debug_nameembedding_76/embeddings/*
dtype0*
shape
:*(
shared_nameembedding_76/embeddings
Г
+embedding_76/embeddings/Read/ReadVariableOpReadVariableOpembedding_76/embeddings*
_output_shapes

:*
dtype0
q
hash_table_4HashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	6435190*
value_dtype0	
q
hash_table_5HashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	6435139*
value_dtype0	
q
hash_table_6HashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	6434864*
value_dtype0	
q
hash_table_7HashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	6434813*
value_dtype0	
q
hash_table_8HashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	6434762*
value_dtype0	
q
hash_table_9HashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	6434711*
value_dtype0	
r
hash_table_10HashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	6434660*
value_dtype0	
r
hash_table_11HashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	6434609*
value_dtype0	
r
hash_table_12HashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	6434558*
value_dtype0	
r
hash_table_13HashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	6434507*
value_dtype0	
r
hash_table_14HashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	6434456*
value_dtype0	
r
hash_table_15HashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	6434405*
value_dtype0	
r
hash_table_16HashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	6434354*
value_dtype0	
v
serving_default_BMIPlaceholder*'
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
|
serving_default_age_rangePlaceholder*'
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
|
serving_default_allergensPlaceholder*'
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
z
serving_default_allergyPlaceholder*'
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
{
serving_default_caloriesPlaceholder*'
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
А
serving_default_carbohydratesPlaceholder*'
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
В
serving_default_clinical_genderPlaceholder*'
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
В
serving_default_cultural_factorPlaceholder*'
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
З
$serving_default_cultural_restrictionPlaceholder*'
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
Й
&serving_default_current_daily_caloriesPlaceholder*'
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
Й
&serving_default_current_working_statusPlaceholder*'
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
}
serving_default_day_numberPlaceholder*'
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€

serving_default_embeddingsPlaceholder*(
_output_shapes
:€€€€€€€€€А*
dtype0*
shape:€€€€€€€€€А
|
serving_default_ethnicityPlaceholder*'
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
v
serving_default_fatPlaceholder*'
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
x
serving_default_fiberPlaceholder*'
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
y
serving_default_heightPlaceholder*'
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
}
serving_default_life_stylePlaceholder*'
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
Б
serving_default_marital_statusPlaceholder*'
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
~
serving_default_meal_type_yPlaceholder*'
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
{
serving_default_next_BMIPlaceholder*'
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
Б
serving_default_nutrition_goalPlaceholder*'
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
М
)serving_default_place_of_meal_consumptionPlaceholder*'
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
x
serving_default_pricePlaceholder*'
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
Л
(serving_default_projected_daily_caloriesPlaceholder*'
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
z
serving_default_proteinPlaceholder*'
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
Ч
4serving_default_social_situation_of_meal_consumptionPlaceholder*'
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
x
serving_default_tastePlaceholder*'
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
Л
(serving_default_time_of_meal_consumptionPlaceholder*'
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
y
serving_default_weightPlaceholder*'
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
в
StatefulPartitionedCallStatefulPartitionedCallserving_default_BMIserving_default_age_rangeserving_default_allergensserving_default_allergyserving_default_caloriesserving_default_carbohydratesserving_default_clinical_genderserving_default_cultural_factor$serving_default_cultural_restriction&serving_default_current_daily_calories&serving_default_current_working_statusserving_default_day_numberserving_default_embeddingsserving_default_ethnicityserving_default_fatserving_default_fiberserving_default_heightserving_default_life_styleserving_default_marital_statusserving_default_meal_type_yserving_default_next_BMIserving_default_nutrition_goal)serving_default_place_of_meal_consumptionserving_default_price(serving_default_projected_daily_caloriesserving_default_protein4serving_default_social_situation_of_meal_consumptionserving_default_taste(serving_default_time_of_meal_consumptionserving_default_weighthash_table_4Const_50hash_table_5Const_49hash_table_6Const_48hash_table_7Const_47hash_table_8Const_46hash_table_9Const_45hash_table_10Const_44hash_table_11Const_43hash_table_12Const_42hash_table_13Const_41hash_table_14Const_40hash_table_15Const_39hash_table_16Const_38hash_table_2Const_37hash_table_3Const_36embedding_90/embeddingsembedding_89/embeddingsembedding_86/embeddingsembedding_85/embeddingsembedding_84/embeddingsembedding_83/embeddingsembedding_82/embeddingsembedding_81/embeddingsembedding_80/embeddingsembedding_79/embeddingsembedding_78/embeddingsembedding_77/embeddingsembedding_76/embeddingsembedding_88/embeddingsembedding_87/embeddingshash_table_1Const_35
hash_tableConst_34user_embedding/kerneluser_embedding/biasfood_embedding/kernelfood_embedding/biascontext_embedding/kernelcontext_embedding/bias&batch_normalization_12/moving_variancebatch_normalization_12/gamma"batch_normalization_12/moving_meanbatch_normalization_12/betafc_layer_0/kernelfc_layer_0/biasfc_layer_1/kernelfc_layer_1/biasfc_layer_2/kernelfc_layer_2/biasoutput_0/kerneloutput_0/bias*l
Tine
c2a																	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*C
_read_only_resource_inputs%
#!<=>?@ABCDEFGHIJOPQRSTUVWXYZ[\]^_`*0
config_proto 

CPU

GPU2*0J 8В *.
f)R'
%__inference_signature_wrapper_7096920
‘
StatefulPartitionedCall_1StatefulPartitionedCallhash_table_16Const_33Const_32*
Tin
2	*
Tout
2*
_collective_manager_ids
 *&
 _has_manual_control_dependencies(*
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *)
f$R"
 __inference__initializer_7097971
‘
StatefulPartitionedCall_2StatefulPartitionedCallhash_table_15Const_30Const_31*
Tin
2	*
Tout
2*
_collective_manager_ids
 *&
 _has_manual_control_dependencies(*
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *)
f$R"
 __inference__initializer_7097986
‘
StatefulPartitionedCall_3StatefulPartitionedCallhash_table_14Const_29Const_28*
Tin
2	*
Tout
2*
_collective_manager_ids
 *&
 _has_manual_control_dependencies(*
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *)
f$R"
 __inference__initializer_7098001
‘
StatefulPartitionedCall_4StatefulPartitionedCallhash_table_13Const_27Const_26*
Tin
2	*
Tout
2*
_collective_manager_ids
 *&
 _has_manual_control_dependencies(*
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *)
f$R"
 __inference__initializer_7098016
‘
StatefulPartitionedCall_5StatefulPartitionedCallhash_table_12Const_25Const_24*
Tin
2	*
Tout
2*
_collective_manager_ids
 *&
 _has_manual_control_dependencies(*
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *)
f$R"
 __inference__initializer_7098031
‘
StatefulPartitionedCall_6StatefulPartitionedCallhash_table_11Const_22Const_23*
Tin
2	*
Tout
2*
_collective_manager_ids
 *&
 _has_manual_control_dependencies(*
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *)
f$R"
 __inference__initializer_7098046
‘
StatefulPartitionedCall_7StatefulPartitionedCallhash_table_10Const_21Const_20*
Tin
2	*
Tout
2*
_collective_manager_ids
 *&
 _has_manual_control_dependencies(*
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *)
f$R"
 __inference__initializer_7098061
”
StatefulPartitionedCall_8StatefulPartitionedCallhash_table_9Const_19Const_18*
Tin
2	*
Tout
2*
_collective_manager_ids
 *&
 _has_manual_control_dependencies(*
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *)
f$R"
 __inference__initializer_7098076
”
StatefulPartitionedCall_9StatefulPartitionedCallhash_table_8Const_17Const_16*
Tin
2	*
Tout
2*
_collective_manager_ids
 *&
 _has_manual_control_dependencies(*
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *)
f$R"
 __inference__initializer_7098091
‘
StatefulPartitionedCall_10StatefulPartitionedCallhash_table_7Const_14Const_15*
Tin
2	*
Tout
2*
_collective_manager_ids
 *&
 _has_manual_control_dependencies(*
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *)
f$R"
 __inference__initializer_7098106
‘
StatefulPartitionedCall_11StatefulPartitionedCallhash_table_6Const_13Const_12*
Tin
2	*
Tout
2*
_collective_manager_ids
 *&
 _has_manual_control_dependencies(*
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *)
f$R"
 __inference__initializer_7098121
‘
StatefulPartitionedCall_12StatefulPartitionedCallhash_table_5Const_11Const_10*
Tin
2	*
Tout
2*
_collective_manager_ids
 *&
 _has_manual_control_dependencies(*
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *)
f$R"
 __inference__initializer_7098136
“
StatefulPartitionedCall_13StatefulPartitionedCallhash_table_4Const_9Const_8*
Tin
2	*
Tout
2*
_collective_manager_ids
 *&
 _has_manual_control_dependencies(*
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *)
f$R"
 __inference__initializer_7098151
“
StatefulPartitionedCall_14StatefulPartitionedCallhash_table_3Const_6Const_7*
Tin
2	*
Tout
2*
_collective_manager_ids
 *&
 _has_manual_control_dependencies(*
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *)
f$R"
 __inference__initializer_7098166
“
StatefulPartitionedCall_15StatefulPartitionedCallhash_table_2Const_5Const_4*
Tin
2	*
Tout
2*
_collective_manager_ids
 *&
 _has_manual_control_dependencies(*
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *)
f$R"
 __inference__initializer_7098181
“
StatefulPartitionedCall_16StatefulPartitionedCallhash_table_1Const_3Const_2*
Tin
2	*
Tout
2*
_collective_manager_ids
 *&
 _has_manual_control_dependencies(*
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *)
f$R"
 __inference__initializer_7098196
ќ
StatefulPartitionedCall_17StatefulPartitionedCall
hash_tableConst_1Const*
Tin
2	*
Tout
2*
_collective_manager_ids
 *&
 _has_manual_control_dependencies(*
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *)
f$R"
 __inference__initializer_7098211
р
NoOpNoOp^StatefulPartitionedCall_1^StatefulPartitionedCall_10^StatefulPartitionedCall_11^StatefulPartitionedCall_12^StatefulPartitionedCall_13^StatefulPartitionedCall_14^StatefulPartitionedCall_15^StatefulPartitionedCall_16^StatefulPartitionedCall_17^StatefulPartitionedCall_2^StatefulPartitionedCall_3^StatefulPartitionedCall_4^StatefulPartitionedCall_5^StatefulPartitionedCall_6^StatefulPartitionedCall_7^StatefulPartitionedCall_8^StatefulPartitionedCall_9
√т
Const_51Const"/device:CPU:0*
_output_shapes
: *
dtype0*ъс
valueпсBлс Bгс
С
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer-16
layer-17
layer-18
layer-19
layer-20
layer-21
layer-22
layer-23
layer-24
layer-25
layer-26
layer-27
layer_with_weights-0
layer-28
layer_with_weights-1
layer-29
layer_with_weights-2
layer-30
 layer_with_weights-3
 layer-31
!layer_with_weights-4
!layer-32
"layer_with_weights-5
"layer-33
#layer_with_weights-6
#layer-34
$layer_with_weights-7
$layer-35
%layer_with_weights-8
%layer-36
&layer_with_weights-9
&layer-37
'layer_with_weights-10
'layer-38
(layer_with_weights-11
(layer-39
)layer_with_weights-12
)layer-40
*layer-41
+layer-42
,layer-43
-layer-44
.layer-45
/layer-46
0layer-47
1layer-48
2layer-49
3layer-50
4layer-51
5layer-52
6layer-53
7layer-54
8layer-55
9layer-56
:layer-57
;layer-58
<layer-59
=layer-60
>layer-61
?layer-62
@layer-63
Alayer-64
Blayer-65
Clayer-66
Dlayer-67
Elayer-68
Flayer-69
Glayer_with_weights-13
Glayer-70
Hlayer_with_weights-14
Hlayer-71
Ilayer-72
Jlayer-73
Klayer-74
Llayer-75
Mlayer-76
Nlayer-77
Olayer-78
Player_with_weights-15
Player-79
Qlayer_with_weights-16
Qlayer-80
Rlayer-81
Slayer_with_weights-17
Slayer-82
Tlayer-83
Ulayer-84
Vlayer_with_weights-18
Vlayer-85
Wlayer_with_weights-19
Wlayer-86
Xlayer_with_weights-20
Xlayer-87
Ylayer_with_weights-21
Ylayer-88
Zlayer_with_weights-22
Zlayer-89
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses
a_default_save_signature
b	optimizer
c
signatures*
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
9
d	keras_api
einput_vocabulary
flookup_table* 
9
g	keras_api
hinput_vocabulary
ilookup_table* 
9
j	keras_api
kinput_vocabulary
llookup_table* 
9
m	keras_api
ninput_vocabulary
olookup_table* 
9
p	keras_api
qinput_vocabulary
rlookup_table* 
9
s	keras_api
tinput_vocabulary
ulookup_table* 
9
v	keras_api
winput_vocabulary
xlookup_table* 
9
y	keras_api
zinput_vocabulary
{lookup_table* 
9
|	keras_api
}input_vocabulary
~lookup_table* 
;
	keras_api
Аinput_vocabulary
Бlookup_table* 
<
В	keras_api
Гinput_vocabulary
Дlookup_table* 
<
Е	keras_api
Жinput_vocabulary
Зlookup_table* 
<
И	keras_api
Йinput_vocabulary
Кlookup_table* 
* 
* 
І
Л	variables
Мtrainable_variables
Нregularization_losses
О	keras_api
П__call__
+Р&call_and_return_all_conditional_losses
С
embeddings*
І
Т	variables
Уtrainable_variables
Фregularization_losses
Х	keras_api
Ц__call__
+Ч&call_and_return_all_conditional_losses
Ш
embeddings*
І
Щ	variables
Ъtrainable_variables
Ыregularization_losses
Ь	keras_api
Э__call__
+Ю&call_and_return_all_conditional_losses
Я
embeddings*
І
†	variables
°trainable_variables
Ґregularization_losses
£	keras_api
§__call__
+•&call_and_return_all_conditional_losses
¶
embeddings*
І
І	variables
®trainable_variables
©regularization_losses
™	keras_api
Ђ__call__
+ђ&call_and_return_all_conditional_losses
≠
embeddings*
І
Ѓ	variables
ѓtrainable_variables
∞regularization_losses
±	keras_api
≤__call__
+≥&call_and_return_all_conditional_losses
і
embeddings*
І
µ	variables
ґtrainable_variables
Јregularization_losses
Є	keras_api
є__call__
+Ї&call_and_return_all_conditional_losses
ї
embeddings*
І
Љ	variables
љtrainable_variables
Њregularization_losses
њ	keras_api
ј__call__
+Ѕ&call_and_return_all_conditional_losses
¬
embeddings*
І
√	variables
ƒtrainable_variables
≈regularization_losses
∆	keras_api
«__call__
+»&call_and_return_all_conditional_losses
…
embeddings*
І
 	variables
Ћtrainable_variables
ћregularization_losses
Ќ	keras_api
ќ__call__
+ѕ&call_and_return_all_conditional_losses
–
embeddings*
І
—	variables
“trainable_variables
”regularization_losses
‘	keras_api
’__call__
+÷&call_and_return_all_conditional_losses
„
embeddings*
І
Ў	variables
ўtrainable_variables
Џregularization_losses
џ	keras_api
№__call__
+Ё&call_and_return_all_conditional_losses
ё
embeddings*
І
я	variables
аtrainable_variables
бregularization_losses
в	keras_api
г__call__
+д&call_and_return_all_conditional_losses
е
embeddings*
* 
<
ж	keras_api
зinput_vocabulary
иlookup_table* 
<
й	keras_api
кinput_vocabulary
лlookup_table* 
Ф
м	variables
нtrainable_variables
оregularization_losses
п	keras_api
р__call__
+с&call_and_return_all_conditional_losses* 
Ф
т	variables
уtrainable_variables
фregularization_losses
х	keras_api
ц__call__
+ч&call_and_return_all_conditional_losses* 
Ф
ш	variables
щtrainable_variables
ъregularization_losses
ы	keras_api
ь__call__
+э&call_and_return_all_conditional_losses* 
Ф
ю	variables
€trainable_variables
Аregularization_losses
Б	keras_api
В__call__
+Г&call_and_return_all_conditional_losses* 
* 
* 
* 
* 
Ф
Д	variables
Еtrainable_variables
Жregularization_losses
З	keras_api
И__call__
+Й&call_and_return_all_conditional_losses* 
Ф
К	variables
Лtrainable_variables
Мregularization_losses
Н	keras_api
О__call__
+П&call_and_return_all_conditional_losses* 
Ф
Р	variables
Сtrainable_variables
Тregularization_losses
У	keras_api
Ф__call__
+Х&call_and_return_all_conditional_losses* 
Ф
Ц	variables
Чtrainable_variables
Шregularization_losses
Щ	keras_api
Ъ__call__
+Ы&call_and_return_all_conditional_losses* 
Ф
Ь	variables
Эtrainable_variables
Юregularization_losses
Я	keras_api
†__call__
+°&call_and_return_all_conditional_losses* 
Ф
Ґ	variables
£trainable_variables
§regularization_losses
•	keras_api
¶__call__
+І&call_and_return_all_conditional_losses* 
Ф
®	variables
©trainable_variables
™regularization_losses
Ђ	keras_api
ђ__call__
+≠&call_and_return_all_conditional_losses* 
Ф
Ѓ	variables
ѓtrainable_variables
∞regularization_losses
±	keras_api
≤__call__
+≥&call_and_return_all_conditional_losses* 
* 
Ф
і	variables
µtrainable_variables
ґregularization_losses
Ј	keras_api
Є__call__
+є&call_and_return_all_conditional_losses* 
<
Ї	keras_api
їinput_vocabulary
Љlookup_table* 
* 
* 
* 
* 
* 
* 
* 
І
љ	variables
Њtrainable_variables
њregularization_losses
ј	keras_api
Ѕ__call__
+¬&call_and_return_all_conditional_losses
√
embeddings*
І
ƒ	variables
≈trainable_variables
∆regularization_losses
«	keras_api
»__call__
+…&call_and_return_all_conditional_losses
 
embeddings*
Ф
Ћ	variables
ћtrainable_variables
Ќregularization_losses
ќ	keras_api
ѕ__call__
+–&call_and_return_all_conditional_losses* 
Ф
—	variables
“trainable_variables
”regularization_losses
‘	keras_api
’__call__
+÷&call_and_return_all_conditional_losses* 
* 
<
„	keras_api
Ўinput_vocabulary
ўlookup_table* 
* 
Ф
Џ	variables
џtrainable_variables
№regularization_losses
Ё	keras_api
ё__call__
+я&call_and_return_all_conditional_losses* 
Ф
а	variables
бtrainable_variables
вregularization_losses
г	keras_api
д__call__
+е&call_and_return_all_conditional_losses* 
Ѓ
ж	variables
зtrainable_variables
иregularization_losses
й	keras_api
к__call__
+л&call_and_return_all_conditional_losses
мkernel
	нbias*
Ѓ
о	variables
пtrainable_variables
рregularization_losses
с	keras_api
т__call__
+у&call_and_return_all_conditional_losses
фkernel
	хbias*
Ф
ц	variables
чtrainable_variables
шregularization_losses
щ	keras_api
ъ__call__
+ы&call_and_return_all_conditional_losses* 
Ѓ
ь	variables
эtrainable_variables
юregularization_losses
€	keras_api
А__call__
+Б&call_and_return_all_conditional_losses
Вkernel
	Гbias*
Ф
Д	variables
Еtrainable_variables
Жregularization_losses
З	keras_api
И__call__
+Й&call_and_return_all_conditional_losses* 
Ф
К	variables
Лtrainable_variables
Мregularization_losses
Н	keras_api
О__call__
+П&call_and_return_all_conditional_losses* 
а
Р	variables
Сtrainable_variables
Тregularization_losses
У	keras_api
Ф__call__
+Х&call_and_return_all_conditional_losses
	Цaxis

Чgamma
	Шbeta
Щmoving_mean
Ъmoving_variance*
Ѓ
Ы	variables
Ьtrainable_variables
Эregularization_losses
Ю	keras_api
Я__call__
+†&call_and_return_all_conditional_losses
°kernel
	Ґbias*
Ѓ
£	variables
§trainable_variables
•regularization_losses
¶	keras_api
І__call__
+®&call_and_return_all_conditional_losses
©kernel
	™bias*
Ѓ
Ђ	variables
ђtrainable_variables
≠regularization_losses
Ѓ	keras_api
ѓ__call__
+∞&call_and_return_all_conditional_losses
±kernel
	≤bias*
Ѓ
≥	variables
іtrainable_variables
µregularization_losses
ґ	keras_api
Ј__call__
+Є&call_and_return_all_conditional_losses
єkernel
	Їbias*
£
С0
Ш1
Я2
¶3
≠4
і5
ї6
¬7
…8
–9
„10
ё11
е12
√13
 14
м15
н16
ф17
х18
В19
Г20
Ч21
Ш22
Щ23
Ъ24
°25
Ґ26
©27
™28
±29
≤30
є31
Ї32*
С
С0
Ш1
Я2
¶3
≠4
і5
ї6
¬7
…8
–9
„10
ё11
е12
√13
 14
м15
н16
ф17
х18
В19
Г20
Ч21
Ш22
°23
Ґ24
©25
™26
±27
≤28
є29
Ї30*
:
ї0
Љ1
љ2
Њ3
њ4
ј5
Ѕ6* 
µ
¬non_trainable_variables
√layers
ƒmetrics
 ≈layer_regularization_losses
∆layer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
a_default_save_signature
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses*

«trace_0
»trace_1* 

…trace_0
 trace_1* 
Ю
Ћ	capture_1
ћ	capture_3
Ќ	capture_5
ќ	capture_7
ѕ	capture_9
–
capture_11
—
capture_13
“
capture_15
”
capture_17
‘
capture_19
’
capture_21
÷
capture_23
„
capture_25
Ў
capture_27
ў
capture_29
Џ
capture_46
џ
capture_48* 
И
№
_variables
Ё_iterations
ё_learning_rate
я_index_dict
а
_momentums
б_velocities
в_update_step_xla*

гserving_default* 
* 
* 
V
д_initializer
е_create_resource
ж_initialize
з_destroy_resource* 
* 
* 
V
и_initializer
й_create_resource
к_initialize
л_destroy_resource* 
* 
* 
V
м_initializer
н_create_resource
о_initialize
п_destroy_resource* 
* 
* 
V
р_initializer
с_create_resource
т_initialize
у_destroy_resource* 
* 
* 
V
ф_initializer
х_create_resource
ц_initialize
ч_destroy_resource* 
* 
* 
V
ш_initializer
щ_create_resource
ъ_initialize
ы_destroy_resource* 
* 
* 
V
ь_initializer
э_create_resource
ю_initialize
€_destroy_resource* 
* 
* 
V
А_initializer
Б_create_resource
В_initialize
Г_destroy_resource* 
* 
* 
V
Д_initializer
Е_create_resource
Ж_initialize
З_destroy_resource* 
* 
* 
V
И_initializer
Й_create_resource
К_initialize
Л_destroy_resource* 
* 
* 
V
М_initializer
Н_create_resource
О_initialize
П_destroy_resource* 
* 
* 
V
Р_initializer
С_create_resource
Т_initialize
У_destroy_resource* 
* 
* 
V
Ф_initializer
Х_create_resource
Ц_initialize
Ч_destroy_resource* 

С0*

С0*
* 
Ю
Шnon_trainable_variables
Щlayers
Ъmetrics
 Ыlayer_regularization_losses
Ьlayer_metrics
Л	variables
Мtrainable_variables
Нregularization_losses
П__call__
+Р&call_and_return_all_conditional_losses
'Р"call_and_return_conditional_losses*

Эtrace_0* 

Юtrace_0* 
ke
VARIABLE_VALUEembedding_76/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE*

Ш0*

Ш0*
* 
Ю
Яnon_trainable_variables
†layers
°metrics
 Ґlayer_regularization_losses
£layer_metrics
Т	variables
Уtrainable_variables
Фregularization_losses
Ц__call__
+Ч&call_and_return_all_conditional_losses
'Ч"call_and_return_conditional_losses*

§trace_0* 

•trace_0* 
ke
VARIABLE_VALUEembedding_77/embeddings:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUE*

Я0*

Я0*
* 
Ю
¶non_trainable_variables
Іlayers
®metrics
 ©layer_regularization_losses
™layer_metrics
Щ	variables
Ъtrainable_variables
Ыregularization_losses
Э__call__
+Ю&call_and_return_all_conditional_losses
'Ю"call_and_return_conditional_losses*

Ђtrace_0* 

ђtrace_0* 
ke
VARIABLE_VALUEembedding_78/embeddings:layer_with_weights-2/embeddings/.ATTRIBUTES/VARIABLE_VALUE*

¶0*

¶0*
* 
Ю
≠non_trainable_variables
Ѓlayers
ѓmetrics
 ∞layer_regularization_losses
±layer_metrics
†	variables
°trainable_variables
Ґregularization_losses
§__call__
+•&call_and_return_all_conditional_losses
'•"call_and_return_conditional_losses*

≤trace_0* 

≥trace_0* 
ke
VARIABLE_VALUEembedding_79/embeddings:layer_with_weights-3/embeddings/.ATTRIBUTES/VARIABLE_VALUE*

≠0*

≠0*
* 
Ю
іnon_trainable_variables
µlayers
ґmetrics
 Јlayer_regularization_losses
Єlayer_metrics
І	variables
®trainable_variables
©regularization_losses
Ђ__call__
+ђ&call_and_return_all_conditional_losses
'ђ"call_and_return_conditional_losses*

єtrace_0* 

Їtrace_0* 
ke
VARIABLE_VALUEembedding_80/embeddings:layer_with_weights-4/embeddings/.ATTRIBUTES/VARIABLE_VALUE*

і0*

і0*
* 
Ю
їnon_trainable_variables
Љlayers
љmetrics
 Њlayer_regularization_losses
њlayer_metrics
Ѓ	variables
ѓtrainable_variables
∞regularization_losses
≤__call__
+≥&call_and_return_all_conditional_losses
'≥"call_and_return_conditional_losses*

јtrace_0* 

Ѕtrace_0* 
ke
VARIABLE_VALUEembedding_81/embeddings:layer_with_weights-5/embeddings/.ATTRIBUTES/VARIABLE_VALUE*

ї0*

ї0*
* 
Ю
¬non_trainable_variables
√layers
ƒmetrics
 ≈layer_regularization_losses
∆layer_metrics
µ	variables
ґtrainable_variables
Јregularization_losses
є__call__
+Ї&call_and_return_all_conditional_losses
'Ї"call_and_return_conditional_losses*

«trace_0* 

»trace_0* 
ke
VARIABLE_VALUEembedding_82/embeddings:layer_with_weights-6/embeddings/.ATTRIBUTES/VARIABLE_VALUE*

¬0*

¬0*
* 
Ю
…non_trainable_variables
 layers
Ћmetrics
 ћlayer_regularization_losses
Ќlayer_metrics
Љ	variables
љtrainable_variables
Њregularization_losses
ј__call__
+Ѕ&call_and_return_all_conditional_losses
'Ѕ"call_and_return_conditional_losses*

ќtrace_0* 

ѕtrace_0* 
ke
VARIABLE_VALUEembedding_83/embeddings:layer_with_weights-7/embeddings/.ATTRIBUTES/VARIABLE_VALUE*

…0*

…0*
* 
Ю
–non_trainable_variables
—layers
“metrics
 ”layer_regularization_losses
‘layer_metrics
√	variables
ƒtrainable_variables
≈regularization_losses
«__call__
+»&call_and_return_all_conditional_losses
'»"call_and_return_conditional_losses*

’trace_0* 

÷trace_0* 
ke
VARIABLE_VALUEembedding_84/embeddings:layer_with_weights-8/embeddings/.ATTRIBUTES/VARIABLE_VALUE*

–0*

–0*
* 
Ю
„non_trainable_variables
Ўlayers
ўmetrics
 Џlayer_regularization_losses
џlayer_metrics
 	variables
Ћtrainable_variables
ћregularization_losses
ќ__call__
+ѕ&call_and_return_all_conditional_losses
'ѕ"call_and_return_conditional_losses*

№trace_0* 

Ёtrace_0* 
ke
VARIABLE_VALUEembedding_85/embeddings:layer_with_weights-9/embeddings/.ATTRIBUTES/VARIABLE_VALUE*

„0*

„0*
* 
Ю
ёnon_trainable_variables
яlayers
аmetrics
 бlayer_regularization_losses
вlayer_metrics
—	variables
“trainable_variables
”regularization_losses
’__call__
+÷&call_and_return_all_conditional_losses
'÷"call_and_return_conditional_losses*

гtrace_0* 

дtrace_0* 
lf
VARIABLE_VALUEembedding_86/embeddings;layer_with_weights-10/embeddings/.ATTRIBUTES/VARIABLE_VALUE*

ё0*

ё0*
* 
Ю
еnon_trainable_variables
жlayers
зmetrics
 иlayer_regularization_losses
йlayer_metrics
Ў	variables
ўtrainable_variables
Џregularization_losses
№__call__
+Ё&call_and_return_all_conditional_losses
'Ё"call_and_return_conditional_losses*

кtrace_0* 

лtrace_0* 
lf
VARIABLE_VALUEembedding_89/embeddings;layer_with_weights-11/embeddings/.ATTRIBUTES/VARIABLE_VALUE*

е0*

е0*
* 
Ю
мnon_trainable_variables
нlayers
оmetrics
 пlayer_regularization_losses
рlayer_metrics
я	variables
аtrainable_variables
бregularization_losses
г__call__
+д&call_and_return_all_conditional_losses
'д"call_and_return_conditional_losses*

сtrace_0* 

тtrace_0* 
lf
VARIABLE_VALUEembedding_90/embeddings;layer_with_weights-12/embeddings/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
V
у_initializer
ф_create_resource
х_initialize
ц_destroy_resource* 
* 
* 
V
ч_initializer
ш_create_resource
щ_initialize
ъ_destroy_resource* 
* 
* 
* 
Ь
ыnon_trainable_variables
ьlayers
эmetrics
 юlayer_regularization_losses
€layer_metrics
м	variables
нtrainable_variables
оregularization_losses
р__call__
+с&call_and_return_all_conditional_losses
'с"call_and_return_conditional_losses* 

Аtrace_0* 

Бtrace_0* 
* 
* 
* 
Ь
Вnon_trainable_variables
Гlayers
Дmetrics
 Еlayer_regularization_losses
Жlayer_metrics
т	variables
уtrainable_variables
фregularization_losses
ц__call__
+ч&call_and_return_all_conditional_losses
'ч"call_and_return_conditional_losses* 

Зtrace_0* 

Иtrace_0* 
* 
* 
* 
Ь
Йnon_trainable_variables
Кlayers
Лmetrics
 Мlayer_regularization_losses
Нlayer_metrics
ш	variables
щtrainable_variables
ъregularization_losses
ь__call__
+э&call_and_return_all_conditional_losses
'э"call_and_return_conditional_losses* 

Оtrace_0* 

Пtrace_0* 
* 
* 
* 
Ь
Рnon_trainable_variables
Сlayers
Тmetrics
 Уlayer_regularization_losses
Фlayer_metrics
ю	variables
€trainable_variables
Аregularization_losses
В__call__
+Г&call_and_return_all_conditional_losses
'Г"call_and_return_conditional_losses* 

Хtrace_0* 

Цtrace_0* 
* 
* 
* 
Ь
Чnon_trainable_variables
Шlayers
Щmetrics
 Ъlayer_regularization_losses
Ыlayer_metrics
Д	variables
Еtrainable_variables
Жregularization_losses
И__call__
+Й&call_and_return_all_conditional_losses
'Й"call_and_return_conditional_losses* 

Ьtrace_0* 

Эtrace_0* 
* 
* 
* 
Ь
Юnon_trainable_variables
Яlayers
†metrics
 °layer_regularization_losses
Ґlayer_metrics
К	variables
Лtrainable_variables
Мregularization_losses
О__call__
+П&call_and_return_all_conditional_losses
'П"call_and_return_conditional_losses* 

£trace_0* 

§trace_0* 
* 
* 
* 
Ь
•non_trainable_variables
¶layers
Іmetrics
 ®layer_regularization_losses
©layer_metrics
Р	variables
Сtrainable_variables
Тregularization_losses
Ф__call__
+Х&call_and_return_all_conditional_losses
'Х"call_and_return_conditional_losses* 

™trace_0* 

Ђtrace_0* 
* 
* 
* 
Ь
ђnon_trainable_variables
≠layers
Ѓmetrics
 ѓlayer_regularization_losses
∞layer_metrics
Ц	variables
Чtrainable_variables
Шregularization_losses
Ъ__call__
+Ы&call_and_return_all_conditional_losses
'Ы"call_and_return_conditional_losses* 

±trace_0* 

≤trace_0* 
* 
* 
* 
Ь
≥non_trainable_variables
іlayers
µmetrics
 ґlayer_regularization_losses
Јlayer_metrics
Ь	variables
Эtrainable_variables
Юregularization_losses
†__call__
+°&call_and_return_all_conditional_losses
'°"call_and_return_conditional_losses* 

Єtrace_0* 

єtrace_0* 
* 
* 
* 
Ь
Їnon_trainable_variables
їlayers
Љmetrics
 љlayer_regularization_losses
Њlayer_metrics
Ґ	variables
£trainable_variables
§regularization_losses
¶__call__
+І&call_and_return_all_conditional_losses
'І"call_and_return_conditional_losses* 

њtrace_0* 

јtrace_0* 
* 
* 
* 
Ь
Ѕnon_trainable_variables
¬layers
√metrics
 ƒlayer_regularization_losses
≈layer_metrics
®	variables
©trainable_variables
™regularization_losses
ђ__call__
+≠&call_and_return_all_conditional_losses
'≠"call_and_return_conditional_losses* 

∆trace_0* 

«trace_0* 
* 
* 
* 
Ь
»non_trainable_variables
…layers
 metrics
 Ћlayer_regularization_losses
ћlayer_metrics
Ѓ	variables
ѓtrainable_variables
∞regularization_losses
≤__call__
+≥&call_and_return_all_conditional_losses
'≥"call_and_return_conditional_losses* 

Ќtrace_0* 

ќtrace_0* 
* 
* 
* 
Ь
ѕnon_trainable_variables
–layers
—metrics
 “layer_regularization_losses
”layer_metrics
і	variables
µtrainable_variables
ґregularization_losses
Є__call__
+є&call_and_return_all_conditional_losses
'є"call_and_return_conditional_losses* 

‘trace_0* 

’trace_0* 
* 
* 
V
÷_initializer
„_create_resource
Ў_initialize
ў_destroy_resource* 

√0*

√0*
* 
Ю
Џnon_trainable_variables
џlayers
№metrics
 Ёlayer_regularization_losses
ёlayer_metrics
љ	variables
Њtrainable_variables
њregularization_losses
Ѕ__call__
+¬&call_and_return_all_conditional_losses
'¬"call_and_return_conditional_losses*

яtrace_0* 

аtrace_0* 
lf
VARIABLE_VALUEembedding_87/embeddings;layer_with_weights-13/embeddings/.ATTRIBUTES/VARIABLE_VALUE*

 0*

 0*
* 
Ю
бnon_trainable_variables
вlayers
гmetrics
 дlayer_regularization_losses
еlayer_metrics
ƒ	variables
≈trainable_variables
∆regularization_losses
»__call__
+…&call_and_return_all_conditional_losses
'…"call_and_return_conditional_losses*

жtrace_0* 

зtrace_0* 
lf
VARIABLE_VALUEembedding_88/embeddings;layer_with_weights-14/embeddings/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
Ь
иnon_trainable_variables
йlayers
кmetrics
 лlayer_regularization_losses
мlayer_metrics
Ћ	variables
ћtrainable_variables
Ќregularization_losses
ѕ__call__
+–&call_and_return_all_conditional_losses
'–"call_and_return_conditional_losses* 

нtrace_0* 

оtrace_0* 
* 
* 
* 
Ь
пnon_trainable_variables
рlayers
сmetrics
 тlayer_regularization_losses
уlayer_metrics
—	variables
“trainable_variables
”regularization_losses
’__call__
+÷&call_and_return_all_conditional_losses
'÷"call_and_return_conditional_losses* 

фtrace_0* 

хtrace_0* 
* 
* 
V
ц_initializer
ч_create_resource
ш_initialize
щ_destroy_resource* 
* 
* 
* 
Ь
ъnon_trainable_variables
ыlayers
ьmetrics
 эlayer_regularization_losses
юlayer_metrics
Џ	variables
џtrainable_variables
№regularization_losses
ё__call__
+я&call_and_return_all_conditional_losses
'я"call_and_return_conditional_losses* 

€trace_0* 

Аtrace_0* 
* 
* 
* 
Ь
Бnon_trainable_variables
Вlayers
Гmetrics
 Дlayer_regularization_losses
Еlayer_metrics
а	variables
бtrainable_variables
вregularization_losses
д__call__
+е&call_and_return_all_conditional_losses
'е"call_and_return_conditional_losses* 

Жtrace_0* 

Зtrace_0* 

м0
н1*

м0
н1*


ї0* 
Ю
Иnon_trainable_variables
Йlayers
Кmetrics
 Лlayer_regularization_losses
Мlayer_metrics
ж	variables
зtrainable_variables
иregularization_losses
к__call__
+л&call_and_return_all_conditional_losses
'л"call_and_return_conditional_losses*

Нtrace_0* 

Оtrace_0* 
f`
VARIABLE_VALUEuser_embedding/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEuser_embedding/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE*

ф0
х1*

ф0
х1*


Љ0* 
Ю
Пnon_trainable_variables
Рlayers
Сmetrics
 Тlayer_regularization_losses
Уlayer_metrics
о	variables
пtrainable_variables
рregularization_losses
т__call__
+у&call_and_return_all_conditional_losses
'у"call_and_return_conditional_losses*

Фtrace_0* 

Хtrace_0* 
f`
VARIABLE_VALUEfood_embedding/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEfood_embedding/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
Ь
Цnon_trainable_variables
Чlayers
Шmetrics
 Щlayer_regularization_losses
Ъlayer_metrics
ц	variables
чtrainable_variables
шregularization_losses
ъ__call__
+ы&call_and_return_all_conditional_losses
'ы"call_and_return_conditional_losses* 

Ыtrace_0* 

Ьtrace_0* 

В0
Г1*

В0
Г1*


љ0* 
Ю
Эnon_trainable_variables
Юlayers
Яmetrics
 †layer_regularization_losses
°layer_metrics
ь	variables
эtrainable_variables
юregularization_losses
А__call__
+Б&call_and_return_all_conditional_losses
'Б"call_and_return_conditional_losses*

Ґtrace_0* 

£trace_0* 
ic
VARIABLE_VALUEcontext_embedding/kernel7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEcontext_embedding/bias5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
Ь
§non_trainable_variables
•layers
¶metrics
 Іlayer_regularization_losses
®layer_metrics
Д	variables
Еtrainable_variables
Жregularization_losses
И__call__
+Й&call_and_return_all_conditional_losses
'Й"call_and_return_conditional_losses* 

©trace_0* 

™trace_0* 
* 
* 
* 
Ь
Ђnon_trainable_variables
ђlayers
≠metrics
 Ѓlayer_regularization_losses
ѓlayer_metrics
К	variables
Лtrainable_variables
Мregularization_losses
О__call__
+П&call_and_return_all_conditional_losses
'П"call_and_return_conditional_losses* 

∞trace_0* 

±trace_0* 
$
Ч0
Ш1
Щ2
Ъ3*

Ч0
Ш1*
* 
Ю
≤non_trainable_variables
≥layers
іmetrics
 µlayer_regularization_losses
ґlayer_metrics
Р	variables
Сtrainable_variables
Тregularization_losses
Ф__call__
+Х&call_and_return_all_conditional_losses
'Х"call_and_return_conditional_losses*

Јtrace_0
Єtrace_1* 

єtrace_0
Їtrace_1* 
* 
lf
VARIABLE_VALUEbatch_normalization_12/gamma6layer_with_weights-18/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_12/beta5layer_with_weights-18/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE"batch_normalization_12/moving_mean<layer_with_weights-18/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUE&batch_normalization_12/moving_variance@layer_with_weights-18/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

°0
Ґ1*

°0
Ґ1*


Њ0* 
Ю
їnon_trainable_variables
Љlayers
љmetrics
 Њlayer_regularization_losses
њlayer_metrics
Ы	variables
Ьtrainable_variables
Эregularization_losses
Я__call__
+†&call_and_return_all_conditional_losses
'†"call_and_return_conditional_losses*

јtrace_0* 

Ѕtrace_0* 
b\
VARIABLE_VALUEfc_layer_0/kernel7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEfc_layer_0/bias5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUE*

©0
™1*

©0
™1*


њ0* 
Ю
¬non_trainable_variables
√layers
ƒmetrics
 ≈layer_regularization_losses
∆layer_metrics
£	variables
§trainable_variables
•regularization_losses
І__call__
+®&call_and_return_all_conditional_losses
'®"call_and_return_conditional_losses*

«trace_0* 

»trace_0* 
b\
VARIABLE_VALUEfc_layer_1/kernel7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEfc_layer_1/bias5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUE*

±0
≤1*

±0
≤1*


ј0* 
Ю
…non_trainable_variables
 layers
Ћmetrics
 ћlayer_regularization_losses
Ќlayer_metrics
Ђ	variables
ђtrainable_variables
≠regularization_losses
ѓ__call__
+∞&call_and_return_all_conditional_losses
'∞"call_and_return_conditional_losses*

ќtrace_0* 

ѕtrace_0* 
b\
VARIABLE_VALUEfc_layer_2/kernel7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEfc_layer_2/bias5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUE*

є0
Ї1*

є0
Ї1*


Ѕ0* 
Ю
–non_trainable_variables
—layers
“metrics
 ”layer_regularization_losses
‘layer_metrics
≥	variables
іtrainable_variables
µregularization_losses
Ј__call__
+Є&call_and_return_all_conditional_losses
'Є"call_and_return_conditional_losses*

’trace_0* 

÷trace_0* 
`Z
VARIABLE_VALUEoutput_0/kernel7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEoutput_0/bias5layer_with_weights-22/bias/.ATTRIBUTES/VARIABLE_VALUE*

„trace_0* 

Ўtrace_0* 

ўtrace_0* 

Џtrace_0* 

џtrace_0* 

№trace_0* 

Ёtrace_0* 

Щ0
Ъ1*
 
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37
'38
(39
)40
*41
+42
,43
-44
.45
/46
047
148
249
350
451
552
653
754
855
956
:57
;58
<59
=60
>61
?62
@63
A64
B65
C66
D67
E68
F69
G70
H71
I72
J73
K74
L75
M76
N77
O78
P79
Q80
R81
S82
T83
U84
V85
W86
X87
Y88
Z89*
$
ё0
я1
а2
б3*
* 
* 
Ю
Ћ	capture_1
ћ	capture_3
Ќ	capture_5
ќ	capture_7
ѕ	capture_9
–
capture_11
—
capture_13
“
capture_15
”
capture_17
‘
capture_19
’
capture_21
÷
capture_23
„
capture_25
Ў
capture_27
ў
capture_29
Џ
capture_46
џ
capture_48* 
Ю
Ћ	capture_1
ћ	capture_3
Ќ	capture_5
ќ	capture_7
ѕ	capture_9
–
capture_11
—
capture_13
“
capture_15
”
capture_17
‘
capture_19
’
capture_21
÷
capture_23
„
capture_25
Ў
capture_27
ў
capture_29
Џ
capture_46
џ
capture_48* 
Ю
Ћ	capture_1
ћ	capture_3
Ќ	capture_5
ќ	capture_7
ѕ	capture_9
–
capture_11
—
capture_13
“
capture_15
”
capture_17
‘
capture_19
’
capture_21
÷
capture_23
„
capture_25
Ў
capture_27
ў
capture_29
Џ
capture_46
џ
capture_48* 
Ю
Ћ	capture_1
ћ	capture_3
Ќ	capture_5
ќ	capture_7
ѕ	capture_9
–
capture_11
—
capture_13
“
capture_15
”
capture_17
‘
capture_19
’
capture_21
÷
capture_23
„
capture_25
Ў
capture_27
ў
capture_29
Џ
capture_46
џ
capture_48* 
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
* 
* 
* 
* 
±
Ё0
в1
г2
д3
е4
ж5
з6
и7
й8
к9
л10
м11
н12
о13
п14
р15
с16
т17
у18
ф19
х20
ц21
ч22
ш23
щ24
ъ25
ы26
ь27
э28
ю29
€30
А31
Б32
В33
Г34
Д35
Е36
Ж37
З38
И39
Й40
К41
Л42
М43
Н44
О45
П46
Р47
С48
Т49
У50
Ф51
Х52
Ц53
Ч54
Ш55
Щ56
Ъ57
Ы58
Ь59
Э60
Ю61
Я62*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
С
в0
д1
ж2
и3
к4
м5
о6
р7
т8
ф9
ц10
ш11
ъ12
ь13
ю14
А15
В16
Д17
Ж18
И19
К20
М21
О22
Р23
Т24
Ф25
Ц26
Ш27
Ъ28
Ь29
Ю30*
С
г0
е1
з2
й3
л4
н5
п6
с7
у8
х9
ч10
щ11
ы12
э13
€14
Б15
Г16
Е17
З18
Й19
Л20
Н21
П22
С23
У24
Х25
Ч26
Щ27
Ы28
Э29
Я30*
…
†trace_0
°trace_1
Ґtrace_2
£trace_3
§trace_4
•trace_5
¶trace_6
Іtrace_7
®trace_8
©trace_9
™trace_10
Ђtrace_11
ђtrace_12
≠trace_13
Ѓtrace_14
ѓtrace_15
∞trace_16
±trace_17
≤trace_18
≥trace_19
іtrace_20
µtrace_21
ґtrace_22
Јtrace_23
Єtrace_24
єtrace_25
Їtrace_26
їtrace_27
Љtrace_28
љtrace_29
Њtrace_30* 
Ю
Ћ	capture_1
ћ	capture_3
Ќ	capture_5
ќ	capture_7
ѕ	capture_9
–
capture_11
—
capture_13
“
capture_15
”
capture_17
‘
capture_19
’
capture_21
÷
capture_23
„
capture_25
Ў
capture_27
ў
capture_29
Џ
capture_46
џ
capture_48* 
* 

њtrace_0* 

јtrace_0* 

Ѕtrace_0* 
* 

¬trace_0* 

√trace_0* 

ƒtrace_0* 
* 

≈trace_0* 

∆trace_0* 

«trace_0* 
* 

»trace_0* 

…trace_0* 

 trace_0* 
* 

Ћtrace_0* 

ћtrace_0* 

Ќtrace_0* 
* 

ќtrace_0* 

ѕtrace_0* 

–trace_0* 
* 

—trace_0* 

“trace_0* 

”trace_0* 
* 

‘trace_0* 

’trace_0* 

÷trace_0* 
* 

„trace_0* 

Ўtrace_0* 

ўtrace_0* 
* 

Џtrace_0* 

џtrace_0* 

№trace_0* 
* 

Ёtrace_0* 

ёtrace_0* 

яtrace_0* 
* 

аtrace_0* 

бtrace_0* 

вtrace_0* 
* 

гtrace_0* 

дtrace_0* 

еtrace_0* 
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
* 

жtrace_0* 

зtrace_0* 

иtrace_0* 
* 

йtrace_0* 

кtrace_0* 

лtrace_0* 
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
* 

мtrace_0* 

нtrace_0* 

оtrace_0* 
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
* 
* 
* 

пtrace_0* 

рtrace_0* 

сtrace_0* 
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
* 
* 
* 
* 


ї0* 
* 
* 
* 
* 
* 
* 


Љ0* 
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


љ0* 
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
* 
* 
* 
* 

Щ0
Ъ1*
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


Њ0* 
* 
* 
* 
* 
* 
* 


њ0* 
* 
* 
* 
* 
* 
* 


ј0* 
* 
* 
* 
* 
* 
* 


Ѕ0* 
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
<
т	variables
у	keras_api

фtotal

хcount*
M
ц	variables
ч	keras_api

шtotal

щcount
ъ
_fn_kwargs*
`
ы	variables
ь	keras_api
э
thresholds
юtrue_positives
€false_positives*
`
А	variables
Б	keras_api
В
thresholds
Гtrue_positives
Дfalse_negatives*
ic
VARIABLE_VALUEAdam/m/embedding_76/embeddings1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEAdam/v/embedding_76/embeddings1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEAdam/m/embedding_77/embeddings1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEAdam/v/embedding_77/embeddings1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEAdam/m/embedding_78/embeddings1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEAdam/v/embedding_78/embeddings1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEAdam/m/embedding_79/embeddings1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEAdam/v/embedding_79/embeddings1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEAdam/m/embedding_80/embeddings1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/v/embedding_80/embeddings2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/m/embedding_81/embeddings2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/v/embedding_81/embeddings2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/m/embedding_82/embeddings2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/v/embedding_82/embeddings2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/m/embedding_83/embeddings2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/v/embedding_83/embeddings2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/m/embedding_84/embeddings2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/v/embedding_84/embeddings2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/m/embedding_85/embeddings2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/v/embedding_85/embeddings2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/m/embedding_86/embeddings2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/v/embedding_86/embeddings2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/m/embedding_89/embeddings2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/v/embedding_89/embeddings2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/m/embedding_90/embeddings2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/v/embedding_90/embeddings2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/m/embedding_87/embeddings2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/v/embedding_87/embeddings2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/m/embedding_88/embeddings2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/v/embedding_88/embeddings2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEAdam/m/user_embedding/kernel2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEAdam/v/user_embedding/kernel2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEAdam/m/user_embedding/bias2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEAdam/v/user_embedding/bias2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEAdam/m/food_embedding/kernel2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEAdam/v/food_embedding/kernel2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEAdam/m/food_embedding/bias2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEAdam/v/food_embedding/bias2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEAdam/m/context_embedding/kernel2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEAdam/v/context_embedding/kernel2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEAdam/m/context_embedding/bias2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEAdam/v/context_embedding/bias2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE#Adam/m/batch_normalization_12/gamma2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE#Adam/v/batch_normalization_12/gamma2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/batch_normalization_12/beta2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/batch_normalization_12/beta2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/m/fc_layer_0/kernel2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/v/fc_layer_0/kernel2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/fc_layer_0/bias2optimizer/_variables/49/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/fc_layer_0/bias2optimizer/_variables/50/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/m/fc_layer_1/kernel2optimizer/_variables/51/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/v/fc_layer_1/kernel2optimizer/_variables/52/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/fc_layer_1/bias2optimizer/_variables/53/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/fc_layer_1/bias2optimizer/_variables/54/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/m/fc_layer_2/kernel2optimizer/_variables/55/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/v/fc_layer_2/kernel2optimizer/_variables/56/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/fc_layer_2/bias2optimizer/_variables/57/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/fc_layer_2/bias2optimizer/_variables/58/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/output_0/kernel2optimizer/_variables/59/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/output_0/kernel2optimizer/_variables/60/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/output_0/bias2optimizer/_variables/61/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/output_0/bias2optimizer/_variables/62/.ATTRIBUTES/VARIABLE_VALUE*
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
* 
* 
* 
* 
* 
* 
"
Е	capture_1
Ж	capture_2* 
* 
* 
"
З	capture_1
И	capture_2* 
* 
* 
"
Й	capture_1
К	capture_2* 
* 
* 
"
Л	capture_1
М	capture_2* 
* 
* 
"
Н	capture_1
О	capture_2* 
* 
* 
"
П	capture_1
Р	capture_2* 
* 
* 
"
С	capture_1
Т	capture_2* 
* 
* 
"
У	capture_1
Ф	capture_2* 
* 
* 
"
Х	capture_1
Ц	capture_2* 
* 
* 
"
Ч	capture_1
Ш	capture_2* 
* 
* 
"
Щ	capture_1
Ъ	capture_2* 
* 
* 
"
Ы	capture_1
Ь	capture_2* 
* 
* 
"
Э	capture_1
Ю	capture_2* 
* 
* 
"
Я	capture_1
†	capture_2* 
* 
* 
"
°	capture_1
Ґ	capture_2* 
* 
* 
"
£	capture_1
§	capture_2* 
* 
* 
"
•	capture_1
¶	capture_2* 
* 

ф0
х1*

т	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

ш0
щ1*

ц	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

ю0
€1*

ы	variables*
* 
ga
VARIABLE_VALUEtrue_positives_1=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEfalse_positives>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUE*

Г0
Д1*

А	variables*
* 
e_
VARIABLE_VALUEtrue_positives=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEfalse_negatives>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUE*
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
* 
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
а
StatefulPartitionedCall_18StatefulPartitionedCallsaver_filenameembedding_76/embeddingsembedding_77/embeddingsembedding_78/embeddingsembedding_79/embeddingsembedding_80/embeddingsembedding_81/embeddingsembedding_82/embeddingsembedding_83/embeddingsembedding_84/embeddingsembedding_85/embeddingsembedding_86/embeddingsembedding_89/embeddingsembedding_90/embeddingsembedding_87/embeddingsembedding_88/embeddingsuser_embedding/kerneluser_embedding/biasfood_embedding/kernelfood_embedding/biascontext_embedding/kernelcontext_embedding/biasbatch_normalization_12/gammabatch_normalization_12/beta"batch_normalization_12/moving_mean&batch_normalization_12/moving_variancefc_layer_0/kernelfc_layer_0/biasfc_layer_1/kernelfc_layer_1/biasfc_layer_2/kernelfc_layer_2/biasoutput_0/kerneloutput_0/bias	iterationlearning_rateAdam/m/embedding_76/embeddingsAdam/v/embedding_76/embeddingsAdam/m/embedding_77/embeddingsAdam/v/embedding_77/embeddingsAdam/m/embedding_78/embeddingsAdam/v/embedding_78/embeddingsAdam/m/embedding_79/embeddingsAdam/v/embedding_79/embeddingsAdam/m/embedding_80/embeddingsAdam/v/embedding_80/embeddingsAdam/m/embedding_81/embeddingsAdam/v/embedding_81/embeddingsAdam/m/embedding_82/embeddingsAdam/v/embedding_82/embeddingsAdam/m/embedding_83/embeddingsAdam/v/embedding_83/embeddingsAdam/m/embedding_84/embeddingsAdam/v/embedding_84/embeddingsAdam/m/embedding_85/embeddingsAdam/v/embedding_85/embeddingsAdam/m/embedding_86/embeddingsAdam/v/embedding_86/embeddingsAdam/m/embedding_89/embeddingsAdam/v/embedding_89/embeddingsAdam/m/embedding_90/embeddingsAdam/v/embedding_90/embeddingsAdam/m/embedding_87/embeddingsAdam/v/embedding_87/embeddingsAdam/m/embedding_88/embeddingsAdam/v/embedding_88/embeddingsAdam/m/user_embedding/kernelAdam/v/user_embedding/kernelAdam/m/user_embedding/biasAdam/v/user_embedding/biasAdam/m/food_embedding/kernelAdam/v/food_embedding/kernelAdam/m/food_embedding/biasAdam/v/food_embedding/biasAdam/m/context_embedding/kernelAdam/v/context_embedding/kernelAdam/m/context_embedding/biasAdam/v/context_embedding/bias#Adam/m/batch_normalization_12/gamma#Adam/v/batch_normalization_12/gamma"Adam/m/batch_normalization_12/beta"Adam/v/batch_normalization_12/betaAdam/m/fc_layer_0/kernelAdam/v/fc_layer_0/kernelAdam/m/fc_layer_0/biasAdam/v/fc_layer_0/biasAdam/m/fc_layer_1/kernelAdam/v/fc_layer_1/kernelAdam/m/fc_layer_1/biasAdam/v/fc_layer_1/biasAdam/m/fc_layer_2/kernelAdam/v/fc_layer_2/kernelAdam/m/fc_layer_2/biasAdam/v/fc_layer_2/biasAdam/m/output_0/kernelAdam/v/output_0/kernelAdam/m/output_0/biasAdam/v/output_0/biastotal_1count_1totalcounttrue_positives_1false_positivestrue_positivesfalse_negativesConst_51*v
Tino
m2k*
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
GPU2*0J 8В *)
f$R"
 __inference__traced_save_7098981
Ў
StatefulPartitionedCall_19StatefulPartitionedCallsaver_filenameembedding_76/embeddingsembedding_77/embeddingsembedding_78/embeddingsembedding_79/embeddingsembedding_80/embeddingsembedding_81/embeddingsembedding_82/embeddingsembedding_83/embeddingsembedding_84/embeddingsembedding_85/embeddingsembedding_86/embeddingsembedding_89/embeddingsembedding_90/embeddingsembedding_87/embeddingsembedding_88/embeddingsuser_embedding/kerneluser_embedding/biasfood_embedding/kernelfood_embedding/biascontext_embedding/kernelcontext_embedding/biasbatch_normalization_12/gammabatch_normalization_12/beta"batch_normalization_12/moving_mean&batch_normalization_12/moving_variancefc_layer_0/kernelfc_layer_0/biasfc_layer_1/kernelfc_layer_1/biasfc_layer_2/kernelfc_layer_2/biasoutput_0/kerneloutput_0/bias	iterationlearning_rateAdam/m/embedding_76/embeddingsAdam/v/embedding_76/embeddingsAdam/m/embedding_77/embeddingsAdam/v/embedding_77/embeddingsAdam/m/embedding_78/embeddingsAdam/v/embedding_78/embeddingsAdam/m/embedding_79/embeddingsAdam/v/embedding_79/embeddingsAdam/m/embedding_80/embeddingsAdam/v/embedding_80/embeddingsAdam/m/embedding_81/embeddingsAdam/v/embedding_81/embeddingsAdam/m/embedding_82/embeddingsAdam/v/embedding_82/embeddingsAdam/m/embedding_83/embeddingsAdam/v/embedding_83/embeddingsAdam/m/embedding_84/embeddingsAdam/v/embedding_84/embeddingsAdam/m/embedding_85/embeddingsAdam/v/embedding_85/embeddingsAdam/m/embedding_86/embeddingsAdam/v/embedding_86/embeddingsAdam/m/embedding_89/embeddingsAdam/v/embedding_89/embeddingsAdam/m/embedding_90/embeddingsAdam/v/embedding_90/embeddingsAdam/m/embedding_87/embeddingsAdam/v/embedding_87/embeddingsAdam/m/embedding_88/embeddingsAdam/v/embedding_88/embeddingsAdam/m/user_embedding/kernelAdam/v/user_embedding/kernelAdam/m/user_embedding/biasAdam/v/user_embedding/biasAdam/m/food_embedding/kernelAdam/v/food_embedding/kernelAdam/m/food_embedding/biasAdam/v/food_embedding/biasAdam/m/context_embedding/kernelAdam/v/context_embedding/kernelAdam/m/context_embedding/biasAdam/v/context_embedding/bias#Adam/m/batch_normalization_12/gamma#Adam/v/batch_normalization_12/gamma"Adam/m/batch_normalization_12/beta"Adam/v/batch_normalization_12/betaAdam/m/fc_layer_0/kernelAdam/v/fc_layer_0/kernelAdam/m/fc_layer_0/biasAdam/v/fc_layer_0/biasAdam/m/fc_layer_1/kernelAdam/v/fc_layer_1/kernelAdam/m/fc_layer_1/biasAdam/v/fc_layer_1/biasAdam/m/fc_layer_2/kernelAdam/v/fc_layer_2/kernelAdam/m/fc_layer_2/biasAdam/v/fc_layer_2/biasAdam/m/output_0/kernelAdam/v/output_0/kernelAdam/m/output_0/biasAdam/v/output_0/biastotal_1count_1totalcounttrue_positives_1false_positivestrue_positivesfalse_negatives*u
Tinn
l2j*
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
GPU2*0J 8В *,
f'R%
#__inference__traced_restore_7099305Й°'
њ
c
G__inference_flatten_78_layer_call_and_return_conditional_losses_7095323

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:€€€€€€€€€X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ь
.
__inference__destroyer_7097975
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
°
І
I__inference_embedding_76_layer_call_and_return_conditional_losses_7097148

inputs	*
embedding_lookup_7097143:
identityИҐembedding_lookupї
embedding_lookupResourceGatherembedding_lookup_7097143inputs*
Tindices0	*+
_class!
loc:@embedding_lookup/7097143*+
_output_shapes
:€€€€€€€€€*
dtype0v
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_output_shapes
:€€€€€€€€€u
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€5
NoOpNoOp^embedding_lookup*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:€€€€€€€€€: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:'#
!
_user_specified_name	7097143
Ѓ
H
,__inference_flatten_85_layer_call_fn_7097432

inputs
identityµ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_flatten_85_layer_call_and_return_conditional_losses_7095372`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Лг
Н6
E__inference_model_12_layer_call_and_return_conditional_losses_7095682
bmi
	age_range
	allergens
allergy
calories
carbohydrates
clinical_gender
cultural_factor
cultural_restriction
current_daily_calories
current_working_status

day_number

embeddings
	ethnicity
fat	
fiber

height

life_style
marital_status
meal_type_y
next_bmi
nutrition_goal
place_of_meal_consumption	
price
projected_daily_calories
protein(
$social_situation_of_meal_consumption	
taste
time_of_meal_consumption

weight@
<string_lookup_107_none_lookup_lookuptablefindv2_table_handleA
=string_lookup_107_none_lookup_lookuptablefindv2_default_value	@
<string_lookup_106_none_lookup_lookuptablefindv2_table_handleA
=string_lookup_106_none_lookup_lookuptablefindv2_default_value	@
<string_lookup_102_none_lookup_lookuptablefindv2_table_handleA
=string_lookup_102_none_lookup_lookuptablefindv2_default_value	@
<string_lookup_101_none_lookup_lookuptablefindv2_table_handleA
=string_lookup_101_none_lookup_lookuptablefindv2_default_value	@
<string_lookup_100_none_lookup_lookuptablefindv2_table_handleA
=string_lookup_100_none_lookup_lookuptablefindv2_default_value	?
;string_lookup_99_none_lookup_lookuptablefindv2_table_handle@
<string_lookup_99_none_lookup_lookuptablefindv2_default_value	?
;string_lookup_98_none_lookup_lookuptablefindv2_table_handle@
<string_lookup_98_none_lookup_lookuptablefindv2_default_value	?
;string_lookup_97_none_lookup_lookuptablefindv2_table_handle@
<string_lookup_97_none_lookup_lookuptablefindv2_default_value	?
;string_lookup_96_none_lookup_lookuptablefindv2_table_handle@
<string_lookup_96_none_lookup_lookuptablefindv2_default_value	?
;string_lookup_95_none_lookup_lookuptablefindv2_table_handle@
<string_lookup_95_none_lookup_lookuptablefindv2_default_value	?
;string_lookup_94_none_lookup_lookuptablefindv2_table_handle@
<string_lookup_94_none_lookup_lookuptablefindv2_default_value	?
;string_lookup_93_none_lookup_lookuptablefindv2_table_handle@
<string_lookup_93_none_lookup_lookuptablefindv2_default_value	?
;string_lookup_92_none_lookup_lookuptablefindv2_table_handle@
<string_lookup_92_none_lookup_lookuptablefindv2_default_value	@
<string_lookup_105_none_lookup_lookuptablefindv2_table_handleA
=string_lookup_105_none_lookup_lookuptablefindv2_default_value	@
<string_lookup_104_none_lookup_lookuptablefindv2_table_handleA
=string_lookup_104_none_lookup_lookuptablefindv2_default_value	&
embedding_90_7095106:W	&
embedding_89_7095117:&
embedding_86_7095128:&
embedding_85_7095139:&
embedding_84_7095150:&
embedding_83_7095161:&
embedding_82_7095172:&
embedding_81_7095183:&
embedding_80_7095194:&
embedding_79_7095205:&
embedding_78_7095216:&
embedding_77_7095227:&
embedding_76_7095238:&
embedding_88_7095249:&
embedding_87_7095260:@
<string_lookup_108_none_lookup_lookuptablefindv2_table_handleA
=string_lookup_108_none_lookup_lookuptablefindv2_default_value	@
<string_lookup_103_none_lookup_lookuptablefindv2_table_handleA
=string_lookup_103_none_lookup_lookuptablefindv2_default_value	)
user_embedding_7095473:	ђ%
user_embedding_7095475:	ђ*
food_embedding_7095492:
£ђ%
food_embedding_7095494:	ђ,
context_embedding_7095522:	Ш'
context_embedding_7095524:-
batch_normalization_12_7095564:	й-
batch_normalization_12_7095566:	й-
batch_normalization_12_7095568:	й-
batch_normalization_12_7095570:	й&
fc_layer_0_7095588:
йА!
fc_layer_0_7095590:	А%
fc_layer_1_7095608:	А@ 
fc_layer_1_7095610:@$
fc_layer_2_7095628:@  
fc_layer_2_7095630: "
output_0_7095648: 
output_0_7095650:
identityИҐ.batch_normalization_12/StatefulPartitionedCallҐ)context_embedding/StatefulPartitionedCallҐ:context_embedding/kernel/Regularizer/L2Loss/ReadVariableOpҐ$embedding_76/StatefulPartitionedCallҐ$embedding_77/StatefulPartitionedCallҐ$embedding_78/StatefulPartitionedCallҐ$embedding_79/StatefulPartitionedCallҐ$embedding_80/StatefulPartitionedCallҐ$embedding_81/StatefulPartitionedCallҐ$embedding_82/StatefulPartitionedCallҐ$embedding_83/StatefulPartitionedCallҐ$embedding_84/StatefulPartitionedCallҐ$embedding_85/StatefulPartitionedCallҐ$embedding_86/StatefulPartitionedCallҐ$embedding_87/StatefulPartitionedCallҐ$embedding_88/StatefulPartitionedCallҐ$embedding_89/StatefulPartitionedCallҐ$embedding_90/StatefulPartitionedCallҐ"fc_layer_0/StatefulPartitionedCallҐ3fc_layer_0/kernel/Regularizer/L2Loss/ReadVariableOpҐ"fc_layer_1/StatefulPartitionedCallҐ3fc_layer_1/kernel/Regularizer/L2Loss/ReadVariableOpҐ"fc_layer_2/StatefulPartitionedCallҐ3fc_layer_2/kernel/Regularizer/L2Loss/ReadVariableOpҐ&food_embedding/StatefulPartitionedCallҐ7food_embedding/kernel/Regularizer/L2Loss/ReadVariableOpҐ output_0/StatefulPartitionedCallҐ1output_0/kernel/Regularizer/L2Loss/ReadVariableOpҐstring_lookup_100/Assert/AssertҐ/string_lookup_100/None_Lookup/LookupTableFindV2Ґstring_lookup_101/Assert/AssertҐ/string_lookup_101/None_Lookup/LookupTableFindV2Ґstring_lookup_102/Assert/AssertҐ/string_lookup_102/None_Lookup/LookupTableFindV2Ґstring_lookup_103/Assert/AssertҐ/string_lookup_103/None_Lookup/LookupTableFindV2Ґstring_lookup_104/Assert/AssertҐ/string_lookup_104/None_Lookup/LookupTableFindV2Ґstring_lookup_105/Assert/AssertҐ/string_lookup_105/None_Lookup/LookupTableFindV2Ґstring_lookup_106/Assert/AssertҐ/string_lookup_106/None_Lookup/LookupTableFindV2Ґstring_lookup_107/Assert/AssertҐ/string_lookup_107/None_Lookup/LookupTableFindV2Ґstring_lookup_108/Assert/AssertҐ/string_lookup_108/None_Lookup/LookupTableFindV2Ґstring_lookup_92/Assert/AssertҐ.string_lookup_92/None_Lookup/LookupTableFindV2Ґstring_lookup_93/Assert/AssertҐ.string_lookup_93/None_Lookup/LookupTableFindV2Ґstring_lookup_94/Assert/AssertҐ.string_lookup_94/None_Lookup/LookupTableFindV2Ґstring_lookup_95/Assert/AssertҐ.string_lookup_95/None_Lookup/LookupTableFindV2Ґstring_lookup_96/Assert/AssertҐ.string_lookup_96/None_Lookup/LookupTableFindV2Ґstring_lookup_97/Assert/AssertҐ.string_lookup_97/None_Lookup/LookupTableFindV2Ґstring_lookup_98/Assert/AssertҐ.string_lookup_98/None_Lookup/LookupTableFindV2Ґstring_lookup_99/Assert/AssertҐ.string_lookup_99/None_Lookup/LookupTableFindV2Ґ&user_embedding/StatefulPartitionedCallҐ7user_embedding/kernel/Regularizer/L2Loss/ReadVariableOpМ
/string_lookup_107/None_Lookup/LookupTableFindV2LookupTableFindV2<string_lookup_107_none_lookup_lookuptablefindv2_table_handle	allergens=string_lookup_107_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:€€€€€€€€€d
string_lookup_107/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€∞
string_lookup_107/EqualEqual8string_lookup_107/None_Lookup/LookupTableFindV2:values:0"string_lookup_107/Equal/y:output:0*
T0	*'
_output_shapes
:€€€€€€€€€f
string_lookup_107/WhereWherestring_lookup_107/Equal:z:0*'
_output_shapes
:€€€€€€€€€Ц
string_lookup_107/GatherNdGatherNd	allergensstring_lookup_107/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:€€€€€€€€€Ф
string_lookup_107/StringFormatStringFormat#string_lookup_107/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*Е
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.`
string_lookup_107/SizeSizestring_lookup_107/Where:index:0*
T0	*
_output_shapes
: ]
string_lookup_107/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : К
string_lookup_107/Equal_1Equalstring_lookup_107/Size:output:0$string_lookup_107/Equal_1/y:output:0*
T0*
_output_shapes
: ї
string_lookup_107/Assert/AssertAssertstring_lookup_107/Equal_1:z:0'string_lookup_107/StringFormat:output:0*

T
2*&
 _has_manual_control_dependencies(*
_output_shapes
 і
string_lookup_107/IdentityIdentity8string_lookup_107/None_Lookup/LookupTableFindV2:values:0 ^string_lookup_107/Assert/Assert*
T0	*'
_output_shapes
:€€€€€€€€€Ч
/string_lookup_106/None_Lookup/LookupTableFindV2LookupTableFindV2<string_lookup_106_none_lookup_lookuptablefindv2_table_handlecultural_restriction=string_lookup_106_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:€€€€€€€€€d
string_lookup_106/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€∞
string_lookup_106/EqualEqual8string_lookup_106/None_Lookup/LookupTableFindV2:values:0"string_lookup_106/Equal/y:output:0*
T0	*'
_output_shapes
:€€€€€€€€€f
string_lookup_106/WhereWherestring_lookup_106/Equal:z:0*'
_output_shapes
:€€€€€€€€€°
string_lookup_106/GatherNdGatherNdcultural_restrictionstring_lookup_106/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:€€€€€€€€€Ф
string_lookup_106/StringFormatStringFormat#string_lookup_106/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*Е
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.`
string_lookup_106/SizeSizestring_lookup_106/Where:index:0*
T0	*
_output_shapes
: ]
string_lookup_106/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : К
string_lookup_106/Equal_1Equalstring_lookup_106/Size:output:0$string_lookup_106/Equal_1/y:output:0*
T0*
_output_shapes
: Ё
string_lookup_106/Assert/AssertAssertstring_lookup_106/Equal_1:z:0'string_lookup_106/StringFormat:output:0 ^string_lookup_107/Assert/Assert*

T
2*&
 _has_manual_control_dependencies(*
_output_shapes
 і
string_lookup_106/IdentityIdentity8string_lookup_106/None_Lookup/LookupTableFindV2:values:0 ^string_lookup_106/Assert/Assert*
T0	*'
_output_shapes
:€€€€€€€€€Л
/string_lookup_102/None_Lookup/LookupTableFindV2LookupTableFindV2<string_lookup_102_none_lookup_lookuptablefindv2_table_handlenext_bmi=string_lookup_102_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:€€€€€€€€€d
string_lookup_102/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€∞
string_lookup_102/EqualEqual8string_lookup_102/None_Lookup/LookupTableFindV2:values:0"string_lookup_102/Equal/y:output:0*
T0	*'
_output_shapes
:€€€€€€€€€f
string_lookup_102/WhereWherestring_lookup_102/Equal:z:0*'
_output_shapes
:€€€€€€€€€Х
string_lookup_102/GatherNdGatherNdnext_bmistring_lookup_102/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:€€€€€€€€€Ф
string_lookup_102/StringFormatStringFormat#string_lookup_102/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*Е
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.`
string_lookup_102/SizeSizestring_lookup_102/Where:index:0*
T0	*
_output_shapes
: ]
string_lookup_102/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : К
string_lookup_102/Equal_1Equalstring_lookup_102/Size:output:0$string_lookup_102/Equal_1/y:output:0*
T0*
_output_shapes
: Ё
string_lookup_102/Assert/AssertAssertstring_lookup_102/Equal_1:z:0'string_lookup_102/StringFormat:output:0 ^string_lookup_106/Assert/Assert*

T
2*&
 _has_manual_control_dependencies(*
_output_shapes
 і
string_lookup_102/IdentityIdentity8string_lookup_102/None_Lookup/LookupTableFindV2:values:0 ^string_lookup_102/Assert/Assert*
T0	*'
_output_shapes
:€€€€€€€€€Ж
/string_lookup_101/None_Lookup/LookupTableFindV2LookupTableFindV2<string_lookup_101_none_lookup_lookuptablefindv2_table_handlebmi=string_lookup_101_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:€€€€€€€€€d
string_lookup_101/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€∞
string_lookup_101/EqualEqual8string_lookup_101/None_Lookup/LookupTableFindV2:values:0"string_lookup_101/Equal/y:output:0*
T0	*'
_output_shapes
:€€€€€€€€€f
string_lookup_101/WhereWherestring_lookup_101/Equal:z:0*'
_output_shapes
:€€€€€€€€€Р
string_lookup_101/GatherNdGatherNdbmistring_lookup_101/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:€€€€€€€€€Ф
string_lookup_101/StringFormatStringFormat#string_lookup_101/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*Е
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.`
string_lookup_101/SizeSizestring_lookup_101/Where:index:0*
T0	*
_output_shapes
: ]
string_lookup_101/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : К
string_lookup_101/Equal_1Equalstring_lookup_101/Size:output:0$string_lookup_101/Equal_1/y:output:0*
T0*
_output_shapes
: Ё
string_lookup_101/Assert/AssertAssertstring_lookup_101/Equal_1:z:0'string_lookup_101/StringFormat:output:0 ^string_lookup_102/Assert/Assert*

T
2*&
 _has_manual_control_dependencies(*
_output_shapes
 і
string_lookup_101/IdentityIdentity8string_lookup_101/None_Lookup/LookupTableFindV2:values:0 ^string_lookup_101/Assert/Assert*
T0	*'
_output_shapes
:€€€€€€€€€М
/string_lookup_100/None_Lookup/LookupTableFindV2LookupTableFindV2<string_lookup_100_none_lookup_lookuptablefindv2_table_handle	ethnicity=string_lookup_100_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:€€€€€€€€€d
string_lookup_100/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€∞
string_lookup_100/EqualEqual8string_lookup_100/None_Lookup/LookupTableFindV2:values:0"string_lookup_100/Equal/y:output:0*
T0	*'
_output_shapes
:€€€€€€€€€f
string_lookup_100/WhereWherestring_lookup_100/Equal:z:0*'
_output_shapes
:€€€€€€€€€Ц
string_lookup_100/GatherNdGatherNd	ethnicitystring_lookup_100/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:€€€€€€€€€Ф
string_lookup_100/StringFormatStringFormat#string_lookup_100/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*Е
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.`
string_lookup_100/SizeSizestring_lookup_100/Where:index:0*
T0	*
_output_shapes
: ]
string_lookup_100/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : К
string_lookup_100/Equal_1Equalstring_lookup_100/Size:output:0$string_lookup_100/Equal_1/y:output:0*
T0*
_output_shapes
: Ё
string_lookup_100/Assert/AssertAssertstring_lookup_100/Equal_1:z:0'string_lookup_100/StringFormat:output:0 ^string_lookup_101/Assert/Assert*

T
2*&
 _has_manual_control_dependencies(*
_output_shapes
 і
string_lookup_100/IdentityIdentity8string_lookup_100/None_Lookup/LookupTableFindV2:values:0 ^string_lookup_100/Assert/Assert*
T0	*'
_output_shapes
:€€€€€€€€€О
.string_lookup_99/None_Lookup/LookupTableFindV2LookupTableFindV2;string_lookup_99_none_lookup_lookuptablefindv2_table_handlemarital_status<string_lookup_99_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:€€€€€€€€€c
string_lookup_99/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€≠
string_lookup_99/EqualEqual7string_lookup_99/None_Lookup/LookupTableFindV2:values:0!string_lookup_99/Equal/y:output:0*
T0	*'
_output_shapes
:€€€€€€€€€d
string_lookup_99/WhereWherestring_lookup_99/Equal:z:0*'
_output_shapes
:€€€€€€€€€Щ
string_lookup_99/GatherNdGatherNdmarital_statusstring_lookup_99/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:€€€€€€€€€Т
string_lookup_99/StringFormatStringFormat"string_lookup_99/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*Е
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.^
string_lookup_99/SizeSizestring_lookup_99/Where:index:0*
T0	*
_output_shapes
: \
string_lookup_99/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : З
string_lookup_99/Equal_1Equalstring_lookup_99/Size:output:0#string_lookup_99/Equal_1/y:output:0*
T0*
_output_shapes
: Џ
string_lookup_99/Assert/AssertAssertstring_lookup_99/Equal_1:z:0&string_lookup_99/StringFormat:output:0 ^string_lookup_100/Assert/Assert*

T
2*&
 _has_manual_control_dependencies(*
_output_shapes
 ±
string_lookup_99/IdentityIdentity7string_lookup_99/None_Lookup/LookupTableFindV2:values:0^string_lookup_99/Assert/Assert*
T0	*'
_output_shapes
:€€€€€€€€€Ц
.string_lookup_98/None_Lookup/LookupTableFindV2LookupTableFindV2;string_lookup_98_none_lookup_lookuptablefindv2_table_handlecurrent_working_status<string_lookup_98_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:€€€€€€€€€c
string_lookup_98/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€≠
string_lookup_98/EqualEqual7string_lookup_98/None_Lookup/LookupTableFindV2:values:0!string_lookup_98/Equal/y:output:0*
T0	*'
_output_shapes
:€€€€€€€€€d
string_lookup_98/WhereWherestring_lookup_98/Equal:z:0*'
_output_shapes
:€€€€€€€€€°
string_lookup_98/GatherNdGatherNdcurrent_working_statusstring_lookup_98/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:€€€€€€€€€Т
string_lookup_98/StringFormatStringFormat"string_lookup_98/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*Е
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.^
string_lookup_98/SizeSizestring_lookup_98/Where:index:0*
T0	*
_output_shapes
: \
string_lookup_98/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : З
string_lookup_98/Equal_1Equalstring_lookup_98/Size:output:0#string_lookup_98/Equal_1/y:output:0*
T0*
_output_shapes
: ў
string_lookup_98/Assert/AssertAssertstring_lookup_98/Equal_1:z:0&string_lookup_98/StringFormat:output:0^string_lookup_99/Assert/Assert*

T
2*&
 _has_manual_control_dependencies(*
_output_shapes
 ±
string_lookup_98/IdentityIdentity7string_lookup_98/None_Lookup/LookupTableFindV2:values:0^string_lookup_98/Assert/Assert*
T0	*'
_output_shapes
:€€€€€€€€€З
.string_lookup_97/None_Lookup/LookupTableFindV2LookupTableFindV2;string_lookup_97_none_lookup_lookuptablefindv2_table_handleallergy<string_lookup_97_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:€€€€€€€€€c
string_lookup_97/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€≠
string_lookup_97/EqualEqual7string_lookup_97/None_Lookup/LookupTableFindV2:values:0!string_lookup_97/Equal/y:output:0*
T0	*'
_output_shapes
:€€€€€€€€€d
string_lookup_97/WhereWherestring_lookup_97/Equal:z:0*'
_output_shapes
:€€€€€€€€€Т
string_lookup_97/GatherNdGatherNdallergystring_lookup_97/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:€€€€€€€€€Т
string_lookup_97/StringFormatStringFormat"string_lookup_97/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*Е
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.^
string_lookup_97/SizeSizestring_lookup_97/Where:index:0*
T0	*
_output_shapes
: \
string_lookup_97/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : З
string_lookup_97/Equal_1Equalstring_lookup_97/Size:output:0#string_lookup_97/Equal_1/y:output:0*
T0*
_output_shapes
: ў
string_lookup_97/Assert/AssertAssertstring_lookup_97/Equal_1:z:0&string_lookup_97/StringFormat:output:0^string_lookup_98/Assert/Assert*

T
2*&
 _has_manual_control_dependencies(*
_output_shapes
 ±
string_lookup_97/IdentityIdentity7string_lookup_97/None_Lookup/LookupTableFindV2:values:0^string_lookup_97/Assert/Assert*
T0	*'
_output_shapes
:€€€€€€€€€П
.string_lookup_96/None_Lookup/LookupTableFindV2LookupTableFindV2;string_lookup_96_none_lookup_lookuptablefindv2_table_handlecultural_factor<string_lookup_96_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:€€€€€€€€€c
string_lookup_96/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€≠
string_lookup_96/EqualEqual7string_lookup_96/None_Lookup/LookupTableFindV2:values:0!string_lookup_96/Equal/y:output:0*
T0	*'
_output_shapes
:€€€€€€€€€d
string_lookup_96/WhereWherestring_lookup_96/Equal:z:0*'
_output_shapes
:€€€€€€€€€Ъ
string_lookup_96/GatherNdGatherNdcultural_factorstring_lookup_96/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:€€€€€€€€€Т
string_lookup_96/StringFormatStringFormat"string_lookup_96/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*Е
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.^
string_lookup_96/SizeSizestring_lookup_96/Where:index:0*
T0	*
_output_shapes
: \
string_lookup_96/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : З
string_lookup_96/Equal_1Equalstring_lookup_96/Size:output:0#string_lookup_96/Equal_1/y:output:0*
T0*
_output_shapes
: ў
string_lookup_96/Assert/AssertAssertstring_lookup_96/Equal_1:z:0&string_lookup_96/StringFormat:output:0^string_lookup_97/Assert/Assert*

T
2*&
 _has_manual_control_dependencies(*
_output_shapes
 ±
string_lookup_96/IdentityIdentity7string_lookup_96/None_Lookup/LookupTableFindV2:values:0^string_lookup_96/Assert/Assert*
T0	*'
_output_shapes
:€€€€€€€€€К
.string_lookup_95/None_Lookup/LookupTableFindV2LookupTableFindV2;string_lookup_95_none_lookup_lookuptablefindv2_table_handle
life_style<string_lookup_95_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:€€€€€€€€€c
string_lookup_95/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€≠
string_lookup_95/EqualEqual7string_lookup_95/None_Lookup/LookupTableFindV2:values:0!string_lookup_95/Equal/y:output:0*
T0	*'
_output_shapes
:€€€€€€€€€d
string_lookup_95/WhereWherestring_lookup_95/Equal:z:0*'
_output_shapes
:€€€€€€€€€Х
string_lookup_95/GatherNdGatherNd
life_stylestring_lookup_95/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:€€€€€€€€€Т
string_lookup_95/StringFormatStringFormat"string_lookup_95/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*Е
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.^
string_lookup_95/SizeSizestring_lookup_95/Where:index:0*
T0	*
_output_shapes
: \
string_lookup_95/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : З
string_lookup_95/Equal_1Equalstring_lookup_95/Size:output:0#string_lookup_95/Equal_1/y:output:0*
T0*
_output_shapes
: ў
string_lookup_95/Assert/AssertAssertstring_lookup_95/Equal_1:z:0&string_lookup_95/StringFormat:output:0^string_lookup_96/Assert/Assert*

T
2*&
 _has_manual_control_dependencies(*
_output_shapes
 ±
string_lookup_95/IdentityIdentity7string_lookup_95/None_Lookup/LookupTableFindV2:values:0^string_lookup_95/Assert/Assert*
T0	*'
_output_shapes
:€€€€€€€€€Й
.string_lookup_94/None_Lookup/LookupTableFindV2LookupTableFindV2;string_lookup_94_none_lookup_lookuptablefindv2_table_handle	age_range<string_lookup_94_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:€€€€€€€€€c
string_lookup_94/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€≠
string_lookup_94/EqualEqual7string_lookup_94/None_Lookup/LookupTableFindV2:values:0!string_lookup_94/Equal/y:output:0*
T0	*'
_output_shapes
:€€€€€€€€€d
string_lookup_94/WhereWherestring_lookup_94/Equal:z:0*'
_output_shapes
:€€€€€€€€€Ф
string_lookup_94/GatherNdGatherNd	age_rangestring_lookup_94/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:€€€€€€€€€Т
string_lookup_94/StringFormatStringFormat"string_lookup_94/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*Е
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.^
string_lookup_94/SizeSizestring_lookup_94/Where:index:0*
T0	*
_output_shapes
: \
string_lookup_94/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : З
string_lookup_94/Equal_1Equalstring_lookup_94/Size:output:0#string_lookup_94/Equal_1/y:output:0*
T0*
_output_shapes
: ў
string_lookup_94/Assert/AssertAssertstring_lookup_94/Equal_1:z:0&string_lookup_94/StringFormat:output:0^string_lookup_95/Assert/Assert*

T
2*&
 _has_manual_control_dependencies(*
_output_shapes
 ±
string_lookup_94/IdentityIdentity7string_lookup_94/None_Lookup/LookupTableFindV2:values:0^string_lookup_94/Assert/Assert*
T0	*'
_output_shapes
:€€€€€€€€€П
.string_lookup_93/None_Lookup/LookupTableFindV2LookupTableFindV2;string_lookup_93_none_lookup_lookuptablefindv2_table_handleclinical_gender<string_lookup_93_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:€€€€€€€€€c
string_lookup_93/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€≠
string_lookup_93/EqualEqual7string_lookup_93/None_Lookup/LookupTableFindV2:values:0!string_lookup_93/Equal/y:output:0*
T0	*'
_output_shapes
:€€€€€€€€€d
string_lookup_93/WhereWherestring_lookup_93/Equal:z:0*'
_output_shapes
:€€€€€€€€€Ъ
string_lookup_93/GatherNdGatherNdclinical_genderstring_lookup_93/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:€€€€€€€€€Т
string_lookup_93/StringFormatStringFormat"string_lookup_93/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*Е
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.^
string_lookup_93/SizeSizestring_lookup_93/Where:index:0*
T0	*
_output_shapes
: \
string_lookup_93/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : З
string_lookup_93/Equal_1Equalstring_lookup_93/Size:output:0#string_lookup_93/Equal_1/y:output:0*
T0*
_output_shapes
: ў
string_lookup_93/Assert/AssertAssertstring_lookup_93/Equal_1:z:0&string_lookup_93/StringFormat:output:0^string_lookup_94/Assert/Assert*

T
2*&
 _has_manual_control_dependencies(*
_output_shapes
 ±
string_lookup_93/IdentityIdentity7string_lookup_93/None_Lookup/LookupTableFindV2:values:0^string_lookup_93/Assert/Assert*
T0	*'
_output_shapes
:€€€€€€€€€О
.string_lookup_92/None_Lookup/LookupTableFindV2LookupTableFindV2;string_lookup_92_none_lookup_lookuptablefindv2_table_handlenutrition_goal<string_lookup_92_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:€€€€€€€€€c
string_lookup_92/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€≠
string_lookup_92/EqualEqual7string_lookup_92/None_Lookup/LookupTableFindV2:values:0!string_lookup_92/Equal/y:output:0*
T0	*'
_output_shapes
:€€€€€€€€€d
string_lookup_92/WhereWherestring_lookup_92/Equal:z:0*'
_output_shapes
:€€€€€€€€€Щ
string_lookup_92/GatherNdGatherNdnutrition_goalstring_lookup_92/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:€€€€€€€€€Т
string_lookup_92/StringFormatStringFormat"string_lookup_92/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*Е
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.^
string_lookup_92/SizeSizestring_lookup_92/Where:index:0*
T0	*
_output_shapes
: \
string_lookup_92/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : З
string_lookup_92/Equal_1Equalstring_lookup_92/Size:output:0#string_lookup_92/Equal_1/y:output:0*
T0*
_output_shapes
: ў
string_lookup_92/Assert/AssertAssertstring_lookup_92/Equal_1:z:0&string_lookup_92/StringFormat:output:0^string_lookup_93/Assert/Assert*

T
2*&
 _has_manual_control_dependencies(*
_output_shapes
 ±
string_lookup_92/IdentityIdentity7string_lookup_92/None_Lookup/LookupTableFindV2:values:0^string_lookup_92/Assert/Assert*
T0	*'
_output_shapes
:€€€€€€€€€І
/string_lookup_105/None_Lookup/LookupTableFindV2LookupTableFindV2<string_lookup_105_none_lookup_lookuptablefindv2_table_handle$social_situation_of_meal_consumption=string_lookup_105_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:€€€€€€€€€d
string_lookup_105/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€∞
string_lookup_105/EqualEqual8string_lookup_105/None_Lookup/LookupTableFindV2:values:0"string_lookup_105/Equal/y:output:0*
T0	*'
_output_shapes
:€€€€€€€€€f
string_lookup_105/WhereWherestring_lookup_105/Equal:z:0*'
_output_shapes
:€€€€€€€€€±
string_lookup_105/GatherNdGatherNd$social_situation_of_meal_consumptionstring_lookup_105/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:€€€€€€€€€Ф
string_lookup_105/StringFormatStringFormat#string_lookup_105/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*Е
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.`
string_lookup_105/SizeSizestring_lookup_105/Where:index:0*
T0	*
_output_shapes
: ]
string_lookup_105/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : К
string_lookup_105/Equal_1Equalstring_lookup_105/Size:output:0$string_lookup_105/Equal_1/y:output:0*
T0*
_output_shapes
: №
string_lookup_105/Assert/AssertAssertstring_lookup_105/Equal_1:z:0'string_lookup_105/StringFormat:output:0^string_lookup_92/Assert/Assert*

T
2*&
 _has_manual_control_dependencies(*
_output_shapes
 і
string_lookup_105/IdentityIdentity8string_lookup_105/None_Lookup/LookupTableFindV2:values:0 ^string_lookup_105/Assert/Assert*
T0	*'
_output_shapes
:€€€€€€€€€Ь
/string_lookup_104/None_Lookup/LookupTableFindV2LookupTableFindV2<string_lookup_104_none_lookup_lookuptablefindv2_table_handleplace_of_meal_consumption=string_lookup_104_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:€€€€€€€€€d
string_lookup_104/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€∞
string_lookup_104/EqualEqual8string_lookup_104/None_Lookup/LookupTableFindV2:values:0"string_lookup_104/Equal/y:output:0*
T0	*'
_output_shapes
:€€€€€€€€€f
string_lookup_104/WhereWherestring_lookup_104/Equal:z:0*'
_output_shapes
:€€€€€€€€€¶
string_lookup_104/GatherNdGatherNdplace_of_meal_consumptionstring_lookup_104/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:€€€€€€€€€Ф
string_lookup_104/StringFormatStringFormat#string_lookup_104/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*Е
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.`
string_lookup_104/SizeSizestring_lookup_104/Where:index:0*
T0	*
_output_shapes
: ]
string_lookup_104/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : К
string_lookup_104/Equal_1Equalstring_lookup_104/Size:output:0$string_lookup_104/Equal_1/y:output:0*
T0*
_output_shapes
: Ё
string_lookup_104/Assert/AssertAssertstring_lookup_104/Equal_1:z:0'string_lookup_104/StringFormat:output:0 ^string_lookup_105/Assert/Assert*

T
2*&
 _has_manual_control_dependencies(*
_output_shapes
 і
string_lookup_104/IdentityIdentity8string_lookup_104/None_Lookup/LookupTableFindV2:values:0 ^string_lookup_104/Assert/Assert*
T0	*'
_output_shapes
:€€€€€€€€€П
$embedding_90/StatefulPartitionedCallStatefulPartitionedCall#string_lookup_107/Identity:output:0embedding_90_7095106*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€	*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_embedding_90_layer_call_and_return_conditional_losses_7095105П
$embedding_89/StatefulPartitionedCallStatefulPartitionedCall#string_lookup_106/Identity:output:0embedding_89_7095117*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_embedding_89_layer_call_and_return_conditional_losses_7095116П
$embedding_86/StatefulPartitionedCallStatefulPartitionedCall#string_lookup_102/Identity:output:0embedding_86_7095128*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_embedding_86_layer_call_and_return_conditional_losses_7095127П
$embedding_85/StatefulPartitionedCallStatefulPartitionedCall#string_lookup_101/Identity:output:0embedding_85_7095139*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_embedding_85_layer_call_and_return_conditional_losses_7095138П
$embedding_84/StatefulPartitionedCallStatefulPartitionedCall#string_lookup_100/Identity:output:0embedding_84_7095150*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_embedding_84_layer_call_and_return_conditional_losses_7095149О
$embedding_83/StatefulPartitionedCallStatefulPartitionedCall"string_lookup_99/Identity:output:0embedding_83_7095161*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_embedding_83_layer_call_and_return_conditional_losses_7095160О
$embedding_82/StatefulPartitionedCallStatefulPartitionedCall"string_lookup_98/Identity:output:0embedding_82_7095172*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_embedding_82_layer_call_and_return_conditional_losses_7095171О
$embedding_81/StatefulPartitionedCallStatefulPartitionedCall"string_lookup_97/Identity:output:0embedding_81_7095183*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_embedding_81_layer_call_and_return_conditional_losses_7095182О
$embedding_80/StatefulPartitionedCallStatefulPartitionedCall"string_lookup_96/Identity:output:0embedding_80_7095194*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_embedding_80_layer_call_and_return_conditional_losses_7095193О
$embedding_79/StatefulPartitionedCallStatefulPartitionedCall"string_lookup_95/Identity:output:0embedding_79_7095205*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_embedding_79_layer_call_and_return_conditional_losses_7095204О
$embedding_78/StatefulPartitionedCallStatefulPartitionedCall"string_lookup_94/Identity:output:0embedding_78_7095216*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_embedding_78_layer_call_and_return_conditional_losses_7095215О
$embedding_77/StatefulPartitionedCallStatefulPartitionedCall"string_lookup_93/Identity:output:0embedding_77_7095227*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_embedding_77_layer_call_and_return_conditional_losses_7095226О
$embedding_76/StatefulPartitionedCallStatefulPartitionedCall"string_lookup_92/Identity:output:0embedding_76_7095238*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_embedding_76_layer_call_and_return_conditional_losses_7095237П
$embedding_88/StatefulPartitionedCallStatefulPartitionedCall#string_lookup_105/Identity:output:0embedding_88_7095249*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_embedding_88_layer_call_and_return_conditional_losses_7095248П
$embedding_87/StatefulPartitionedCallStatefulPartitionedCall#string_lookup_104/Identity:output:0embedding_87_7095260*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_embedding_87_layer_call_and_return_conditional_losses_7095259з
flatten_89/PartitionedCallPartitionedCall-embedding_89/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_flatten_89_layer_call_and_return_conditional_losses_7095268з
flatten_90/PartitionedCallPartitionedCall-embedding_90/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_flatten_90_layer_call_and_return_conditional_losses_7095275И
/string_lookup_108/None_Lookup/LookupTableFindV2LookupTableFindV2<string_lookup_108_none_lookup_lookuptablefindv2_table_handletaste=string_lookup_108_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:€€€€€€€€€d
string_lookup_108/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€∞
string_lookup_108/EqualEqual8string_lookup_108/None_Lookup/LookupTableFindV2:values:0"string_lookup_108/Equal/y:output:0*
T0	*'
_output_shapes
:€€€€€€€€€f
string_lookup_108/WhereWherestring_lookup_108/Equal:z:0*'
_output_shapes
:€€€€€€€€€Т
string_lookup_108/GatherNdGatherNdtastestring_lookup_108/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:€€€€€€€€€Ф
string_lookup_108/StringFormatStringFormat#string_lookup_108/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*Е
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.`
string_lookup_108/SizeSizestring_lookup_108/Where:index:0*
T0	*
_output_shapes
: ]
string_lookup_108/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : К
string_lookup_108/Equal_1Equalstring_lookup_108/Size:output:0$string_lookup_108/Equal_1/y:output:0*
T0*
_output_shapes
: Ё
string_lookup_108/Assert/AssertAssertstring_lookup_108/Equal_1:z:0'string_lookup_108/StringFormat:output:0 ^string_lookup_104/Assert/Assert*

T
2*&
 _has_manual_control_dependencies(*
_output_shapes
 і
string_lookup_108/IdentityIdentity8string_lookup_108/None_Lookup/LookupTableFindV2:values:0 ^string_lookup_108/Assert/Assert*
T0	*'
_output_shapes
:€€€€€€€€€m
string_lookup_108/bincount/SizeSize#string_lookup_108/Identity:output:0*
T0	*
_output_shapes
: f
$string_lookup_108/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : І
"string_lookup_108/bincount/GreaterGreater(string_lookup_108/bincount/Size:output:0-string_lookup_108/bincount/Greater/y:output:0*
T0*
_output_shapes
: 
string_lookup_108/bincount/CastCast&string_lookup_108/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: q
 string_lookup_108/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Ц
string_lookup_108/bincount/MaxMax#string_lookup_108/Identity:output:0)string_lookup_108/bincount/Const:output:0*
T0	*
_output_shapes
: b
 string_lookup_108/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 RЬ
string_lookup_108/bincount/addAddV2'string_lookup_108/bincount/Max:output:0)string_lookup_108/bincount/add/y:output:0*
T0	*
_output_shapes
: П
string_lookup_108/bincount/mulMul#string_lookup_108/bincount/Cast:y:0"string_lookup_108/bincount/add:z:0*
T0	*
_output_shapes
: f
$string_lookup_108/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R°
"string_lookup_108/bincount/MaximumMaximum-string_lookup_108/bincount/minlength:output:0"string_lookup_108/bincount/mul:z:0*
T0	*
_output_shapes
: f
$string_lookup_108/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R•
"string_lookup_108/bincount/MinimumMinimum-string_lookup_108/bincount/maxlength:output:0&string_lookup_108/bincount/Maximum:z:0*
T0	*
_output_shapes
: e
"string_lookup_108/bincount/Const_1Const*
_output_shapes
: *
dtype0*
valueB Ж
(string_lookup_108/bincount/DenseBincountDenseBincount#string_lookup_108/Identity:output:0&string_lookup_108/bincount/Minimum:z:0+string_lookup_108/bincount/Const_1:output:0*
T0*

Tidx0	*'
_output_shapes
:€€€€€€€€€*
binary_output(з
flatten_76/PartitionedCallPartitionedCall-embedding_76/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_flatten_76_layer_call_and_return_conditional_losses_7095309з
flatten_77/PartitionedCallPartitionedCall-embedding_77/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_flatten_77_layer_call_and_return_conditional_losses_7095316з
flatten_78/PartitionedCallPartitionedCall-embedding_78/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_flatten_78_layer_call_and_return_conditional_losses_7095323з
flatten_79/PartitionedCallPartitionedCall-embedding_79/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_flatten_79_layer_call_and_return_conditional_losses_7095330з
flatten_80/PartitionedCallPartitionedCall-embedding_80/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_flatten_80_layer_call_and_return_conditional_losses_7095337з
flatten_81/PartitionedCallPartitionedCall-embedding_81/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_flatten_81_layer_call_and_return_conditional_losses_7095344з
flatten_82/PartitionedCallPartitionedCall-embedding_82/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_flatten_82_layer_call_and_return_conditional_losses_7095351з
flatten_83/PartitionedCallPartitionedCall-embedding_83/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_flatten_83_layer_call_and_return_conditional_losses_7095358з
flatten_84/PartitionedCallPartitionedCall-embedding_84/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_flatten_84_layer_call_and_return_conditional_losses_7095365з
flatten_85/PartitionedCallPartitionedCall-embedding_85/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_flatten_85_layer_call_and_return_conditional_losses_7095372з
flatten_86/PartitionedCallPartitionedCall-embedding_86/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_flatten_86_layer_call_and_return_conditional_losses_7095379О
/string_lookup_103/None_Lookup/LookupTableFindV2LookupTableFindV2<string_lookup_103_none_lookup_lookuptablefindv2_table_handlemeal_type_y=string_lookup_103_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:€€€€€€€€€d
string_lookup_103/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€∞
string_lookup_103/EqualEqual8string_lookup_103/None_Lookup/LookupTableFindV2:values:0"string_lookup_103/Equal/y:output:0*
T0	*'
_output_shapes
:€€€€€€€€€f
string_lookup_103/WhereWherestring_lookup_103/Equal:z:0*'
_output_shapes
:€€€€€€€€€Ш
string_lookup_103/GatherNdGatherNdmeal_type_ystring_lookup_103/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:€€€€€€€€€Ф
string_lookup_103/StringFormatStringFormat#string_lookup_103/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*Е
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.`
string_lookup_103/SizeSizestring_lookup_103/Where:index:0*
T0	*
_output_shapes
: ]
string_lookup_103/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : К
string_lookup_103/Equal_1Equalstring_lookup_103/Size:output:0$string_lookup_103/Equal_1/y:output:0*
T0*
_output_shapes
: Ё
string_lookup_103/Assert/AssertAssertstring_lookup_103/Equal_1:z:0'string_lookup_103/StringFormat:output:0 ^string_lookup_108/Assert/Assert*

T
2*&
 _has_manual_control_dependencies(*
_output_shapes
 і
string_lookup_103/IdentityIdentity8string_lookup_103/None_Lookup/LookupTableFindV2:values:0 ^string_lookup_103/Assert/Assert*
T0	*'
_output_shapes
:€€€€€€€€€m
string_lookup_103/bincount/SizeSize#string_lookup_103/Identity:output:0*
T0	*
_output_shapes
: f
$string_lookup_103/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : І
"string_lookup_103/bincount/GreaterGreater(string_lookup_103/bincount/Size:output:0-string_lookup_103/bincount/Greater/y:output:0*
T0*
_output_shapes
: 
string_lookup_103/bincount/CastCast&string_lookup_103/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: q
 string_lookup_103/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Ц
string_lookup_103/bincount/MaxMax#string_lookup_103/Identity:output:0)string_lookup_103/bincount/Const:output:0*
T0	*
_output_shapes
: b
 string_lookup_103/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 RЬ
string_lookup_103/bincount/addAddV2'string_lookup_103/bincount/Max:output:0)string_lookup_103/bincount/add/y:output:0*
T0	*
_output_shapes
: П
string_lookup_103/bincount/mulMul#string_lookup_103/bincount/Cast:y:0"string_lookup_103/bincount/add:z:0*
T0	*
_output_shapes
: g
$string_lookup_103/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value
B	 RУ°
"string_lookup_103/bincount/MaximumMaximum-string_lookup_103/bincount/minlength:output:0"string_lookup_103/bincount/mul:z:0*
T0	*
_output_shapes
: g
$string_lookup_103/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
value
B	 RУ•
"string_lookup_103/bincount/MinimumMinimum-string_lookup_103/bincount/maxlength:output:0&string_lookup_103/bincount/Maximum:z:0*
T0	*
_output_shapes
: e
"string_lookup_103/bincount/Const_1Const*
_output_shapes
: *
dtype0*
valueB З
(string_lookup_103/bincount/DenseBincountDenseBincount#string_lookup_103/Identity:output:0&string_lookup_103/bincount/Minimum:z:0+string_lookup_103/bincount/Const_1:output:0*
T0*

Tidx0	*(
_output_shapes
:€€€€€€€€€У*
binary_output(з
flatten_87/PartitionedCallPartitionedCall-embedding_87/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_flatten_87_layer_call_and_return_conditional_losses_7095413з
flatten_88/PartitionedCallPartitionedCall-embedding_88/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_flatten_88_layer_call_and_return_conditional_losses_7095420И
concatenate_33/PartitionedCallPartitionedCall#flatten_89/PartitionedCall:output:0calories#flatten_90/PartitionedCall:output:01string_lookup_108/bincount/DenseBincount:output:0pricefiberfatproteincarbohydrates
embeddings*
Tin
2
*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€£* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_concatenate_33_layer_call_and_return_conditional_losses_7095436І
concatenate_31/PartitionedCallPartitionedCall#flatten_76/PartitionedCall:output:0#flatten_77/PartitionedCall:output:0#flatten_78/PartitionedCall:output:0#flatten_79/PartitionedCall:output:0weightheightprojected_daily_caloriescurrent_daily_calories#flatten_80/PartitionedCall:output:0#flatten_81/PartitionedCall:output:0#flatten_82/PartitionedCall:output:0#flatten_83/PartitionedCall:output:0#flatten_84/PartitionedCall:output:0#flatten_85/PartitionedCall:output:0#flatten_86/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_concatenate_31_layer_call_and_return_conditional_losses_7095457∞
&user_embedding/StatefulPartitionedCallStatefulPartitionedCall'concatenate_31/PartitionedCall:output:0user_embedding_7095473user_embedding_7095475*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_user_embedding_layer_call_and_return_conditional_losses_7095472∞
&food_embedding/StatefulPartitionedCallStatefulPartitionedCall'concatenate_33/PartitionedCall:output:0food_embedding_7095492food_embedding_7095494*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_food_embedding_layer_call_and_return_conditional_losses_7095491и
concatenate_32/PartitionedCallPartitionedCall
day_number1string_lookup_103/bincount/DenseBincount:output:0time_of_meal_consumption#flatten_87/PartitionedCall:output:0#flatten_88/PartitionedCall:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€Ш* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_concatenate_32_layer_call_and_return_conditional_losses_7095506ї
)context_embedding/StatefulPartitionedCallStatefulPartitionedCall'concatenate_32/PartitionedCall:output:0context_embedding_7095522context_embedding_7095524*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_context_embedding_layer_call_and_return_conditional_losses_7095521У
dot_12/PartitionedCallPartitionedCall/user_embedding/StatefulPartitionedCall:output:0/food_embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dot_12_layer_call_and_return_conditional_losses_7095552ы
concatenate_34/PartitionedCallPartitionedCall/user_embedding/StatefulPartitionedCall:output:0/food_embedding/StatefulPartitionedCall:output:02context_embedding/StatefulPartitionedCall:output:0dot_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€й* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_concatenate_34_layer_call_and_return_conditional_losses_7095562Т
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall'concatenate_34/PartitionedCall:output:0batch_normalization_12_7095564batch_normalization_12_7095566batch_normalization_12_7095568batch_normalization_12_7095570*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€й*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_7094840∞
"fc_layer_0/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0fc_layer_0_7095588fc_layer_0_7095590*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_fc_layer_0_layer_call_and_return_conditional_losses_7095587£
"fc_layer_1/StatefulPartitionedCallStatefulPartitionedCall+fc_layer_0/StatefulPartitionedCall:output:0fc_layer_1_7095608fc_layer_1_7095610*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_fc_layer_1_layer_call_and_return_conditional_losses_7095607£
"fc_layer_2/StatefulPartitionedCallStatefulPartitionedCall+fc_layer_1/StatefulPartitionedCall:output:0fc_layer_2_7095628fc_layer_2_7095630*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_fc_layer_2_layer_call_and_return_conditional_losses_7095627Ы
 output_0/StatefulPartitionedCallStatefulPartitionedCall+fc_layer_2/StatefulPartitionedCall:output:0output_0_7095648output_0_7095650*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_output_0_layer_call_and_return_conditional_losses_7095647П
7user_embedding/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpuser_embedding_7095473*
_output_shapes
:	ђ*
dtype0Ф
(user_embedding/kernel/Regularizer/L2LossL2Loss?user_embedding/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: l
'user_embedding/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„£<≤
%user_embedding/kernel/Regularizer/mulMul0user_embedding/kernel/Regularizer/mul/x:output:01user_embedding/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Р
7food_embedding/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpfood_embedding_7095492* 
_output_shapes
:
£ђ*
dtype0Ф
(food_embedding/kernel/Regularizer/L2LossL2Loss?food_embedding/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: l
'food_embedding/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„£<≤
%food_embedding/kernel/Regularizer/mulMul0food_embedding/kernel/Regularizer/mul/x:output:01food_embedding/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Х
:context_embedding/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpcontext_embedding_7095522*
_output_shapes
:	Ш*
dtype0Ъ
+context_embedding/kernel/Regularizer/L2LossL2LossBcontext_embedding/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: o
*context_embedding/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„£<ї
(context_embedding/kernel/Regularizer/mulMul3context_embedding/kernel/Regularizer/mul/x:output:04context_embedding/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: И
3fc_layer_0/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpfc_layer_0_7095588* 
_output_shapes
:
йА*
dtype0М
$fc_layer_0/kernel/Regularizer/L2LossL2Loss;fc_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#fc_layer_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„£<¶
!fc_layer_0/kernel/Regularizer/mulMul,fc_layer_0/kernel/Regularizer/mul/x:output:0-fc_layer_0/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: З
3fc_layer_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpfc_layer_1_7095608*
_output_shapes
:	А@*
dtype0М
$fc_layer_1/kernel/Regularizer/L2LossL2Loss;fc_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#fc_layer_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„£<¶
!fc_layer_1/kernel/Regularizer/mulMul,fc_layer_1/kernel/Regularizer/mul/x:output:0-fc_layer_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Ж
3fc_layer_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpfc_layer_2_7095628*
_output_shapes

:@ *
dtype0М
$fc_layer_2/kernel/Regularizer/L2LossL2Loss;fc_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#fc_layer_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„£<¶
!fc_layer_2/kernel/Regularizer/mulMul,fc_layer_2/kernel/Regularizer/mul/x:output:0-fc_layer_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: В
1output_0/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpoutput_0_7095648*
_output_shapes

: *
dtype0И
"output_0/kernel/Regularizer/L2LossL2Loss9output_0/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!output_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„£<†
output_0/kernel/Regularizer/mulMul*output_0/kernel/Regularizer/mul/x:output:0+output_0/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: x
IdentityIdentity)output_0/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€Ј
NoOpNoOp/^batch_normalization_12/StatefulPartitionedCall*^context_embedding/StatefulPartitionedCall;^context_embedding/kernel/Regularizer/L2Loss/ReadVariableOp%^embedding_76/StatefulPartitionedCall%^embedding_77/StatefulPartitionedCall%^embedding_78/StatefulPartitionedCall%^embedding_79/StatefulPartitionedCall%^embedding_80/StatefulPartitionedCall%^embedding_81/StatefulPartitionedCall%^embedding_82/StatefulPartitionedCall%^embedding_83/StatefulPartitionedCall%^embedding_84/StatefulPartitionedCall%^embedding_85/StatefulPartitionedCall%^embedding_86/StatefulPartitionedCall%^embedding_87/StatefulPartitionedCall%^embedding_88/StatefulPartitionedCall%^embedding_89/StatefulPartitionedCall%^embedding_90/StatefulPartitionedCall#^fc_layer_0/StatefulPartitionedCall4^fc_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp#^fc_layer_1/StatefulPartitionedCall4^fc_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp#^fc_layer_2/StatefulPartitionedCall4^fc_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp'^food_embedding/StatefulPartitionedCall8^food_embedding/kernel/Regularizer/L2Loss/ReadVariableOp!^output_0/StatefulPartitionedCall2^output_0/kernel/Regularizer/L2Loss/ReadVariableOp ^string_lookup_100/Assert/Assert0^string_lookup_100/None_Lookup/LookupTableFindV2 ^string_lookup_101/Assert/Assert0^string_lookup_101/None_Lookup/LookupTableFindV2 ^string_lookup_102/Assert/Assert0^string_lookup_102/None_Lookup/LookupTableFindV2 ^string_lookup_103/Assert/Assert0^string_lookup_103/None_Lookup/LookupTableFindV2 ^string_lookup_104/Assert/Assert0^string_lookup_104/None_Lookup/LookupTableFindV2 ^string_lookup_105/Assert/Assert0^string_lookup_105/None_Lookup/LookupTableFindV2 ^string_lookup_106/Assert/Assert0^string_lookup_106/None_Lookup/LookupTableFindV2 ^string_lookup_107/Assert/Assert0^string_lookup_107/None_Lookup/LookupTableFindV2 ^string_lookup_108/Assert/Assert0^string_lookup_108/None_Lookup/LookupTableFindV2^string_lookup_92/Assert/Assert/^string_lookup_92/None_Lookup/LookupTableFindV2^string_lookup_93/Assert/Assert/^string_lookup_93/None_Lookup/LookupTableFindV2^string_lookup_94/Assert/Assert/^string_lookup_94/None_Lookup/LookupTableFindV2^string_lookup_95/Assert/Assert/^string_lookup_95/None_Lookup/LookupTableFindV2^string_lookup_96/Assert/Assert/^string_lookup_96/None_Lookup/LookupTableFindV2^string_lookup_97/Assert/Assert/^string_lookup_97/None_Lookup/LookupTableFindV2^string_lookup_98/Assert/Assert/^string_lookup_98/None_Lookup/LookupTableFindV2^string_lookup_99/Assert/Assert/^string_lookup_99/None_Lookup/LookupTableFindV2'^user_embedding/StatefulPartitionedCall8^user_embedding/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*÷
_input_shapesƒ
Ѕ:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€А:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2V
)context_embedding/StatefulPartitionedCall)context_embedding/StatefulPartitionedCall2x
:context_embedding/kernel/Regularizer/L2Loss/ReadVariableOp:context_embedding/kernel/Regularizer/L2Loss/ReadVariableOp2L
$embedding_76/StatefulPartitionedCall$embedding_76/StatefulPartitionedCall2L
$embedding_77/StatefulPartitionedCall$embedding_77/StatefulPartitionedCall2L
$embedding_78/StatefulPartitionedCall$embedding_78/StatefulPartitionedCall2L
$embedding_79/StatefulPartitionedCall$embedding_79/StatefulPartitionedCall2L
$embedding_80/StatefulPartitionedCall$embedding_80/StatefulPartitionedCall2L
$embedding_81/StatefulPartitionedCall$embedding_81/StatefulPartitionedCall2L
$embedding_82/StatefulPartitionedCall$embedding_82/StatefulPartitionedCall2L
$embedding_83/StatefulPartitionedCall$embedding_83/StatefulPartitionedCall2L
$embedding_84/StatefulPartitionedCall$embedding_84/StatefulPartitionedCall2L
$embedding_85/StatefulPartitionedCall$embedding_85/StatefulPartitionedCall2L
$embedding_86/StatefulPartitionedCall$embedding_86/StatefulPartitionedCall2L
$embedding_87/StatefulPartitionedCall$embedding_87/StatefulPartitionedCall2L
$embedding_88/StatefulPartitionedCall$embedding_88/StatefulPartitionedCall2L
$embedding_89/StatefulPartitionedCall$embedding_89/StatefulPartitionedCall2L
$embedding_90/StatefulPartitionedCall$embedding_90/StatefulPartitionedCall2H
"fc_layer_0/StatefulPartitionedCall"fc_layer_0/StatefulPartitionedCall2j
3fc_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp3fc_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp2H
"fc_layer_1/StatefulPartitionedCall"fc_layer_1/StatefulPartitionedCall2j
3fc_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp3fc_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp2H
"fc_layer_2/StatefulPartitionedCall"fc_layer_2/StatefulPartitionedCall2j
3fc_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp3fc_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp2P
&food_embedding/StatefulPartitionedCall&food_embedding/StatefulPartitionedCall2r
7food_embedding/kernel/Regularizer/L2Loss/ReadVariableOp7food_embedding/kernel/Regularizer/L2Loss/ReadVariableOp2D
 output_0/StatefulPartitionedCall output_0/StatefulPartitionedCall2f
1output_0/kernel/Regularizer/L2Loss/ReadVariableOp1output_0/kernel/Regularizer/L2Loss/ReadVariableOp2B
string_lookup_100/Assert/Assertstring_lookup_100/Assert/Assert2b
/string_lookup_100/None_Lookup/LookupTableFindV2/string_lookup_100/None_Lookup/LookupTableFindV22B
string_lookup_101/Assert/Assertstring_lookup_101/Assert/Assert2b
/string_lookup_101/None_Lookup/LookupTableFindV2/string_lookup_101/None_Lookup/LookupTableFindV22B
string_lookup_102/Assert/Assertstring_lookup_102/Assert/Assert2b
/string_lookup_102/None_Lookup/LookupTableFindV2/string_lookup_102/None_Lookup/LookupTableFindV22B
string_lookup_103/Assert/Assertstring_lookup_103/Assert/Assert2b
/string_lookup_103/None_Lookup/LookupTableFindV2/string_lookup_103/None_Lookup/LookupTableFindV22B
string_lookup_104/Assert/Assertstring_lookup_104/Assert/Assert2b
/string_lookup_104/None_Lookup/LookupTableFindV2/string_lookup_104/None_Lookup/LookupTableFindV22B
string_lookup_105/Assert/Assertstring_lookup_105/Assert/Assert2b
/string_lookup_105/None_Lookup/LookupTableFindV2/string_lookup_105/None_Lookup/LookupTableFindV22B
string_lookup_106/Assert/Assertstring_lookup_106/Assert/Assert2b
/string_lookup_106/None_Lookup/LookupTableFindV2/string_lookup_106/None_Lookup/LookupTableFindV22B
string_lookup_107/Assert/Assertstring_lookup_107/Assert/Assert2b
/string_lookup_107/None_Lookup/LookupTableFindV2/string_lookup_107/None_Lookup/LookupTableFindV22B
string_lookup_108/Assert/Assertstring_lookup_108/Assert/Assert2b
/string_lookup_108/None_Lookup/LookupTableFindV2/string_lookup_108/None_Lookup/LookupTableFindV22@
string_lookup_92/Assert/Assertstring_lookup_92/Assert/Assert2`
.string_lookup_92/None_Lookup/LookupTableFindV2.string_lookup_92/None_Lookup/LookupTableFindV22@
string_lookup_93/Assert/Assertstring_lookup_93/Assert/Assert2`
.string_lookup_93/None_Lookup/LookupTableFindV2.string_lookup_93/None_Lookup/LookupTableFindV22@
string_lookup_94/Assert/Assertstring_lookup_94/Assert/Assert2`
.string_lookup_94/None_Lookup/LookupTableFindV2.string_lookup_94/None_Lookup/LookupTableFindV22@
string_lookup_95/Assert/Assertstring_lookup_95/Assert/Assert2`
.string_lookup_95/None_Lookup/LookupTableFindV2.string_lookup_95/None_Lookup/LookupTableFindV22@
string_lookup_96/Assert/Assertstring_lookup_96/Assert/Assert2`
.string_lookup_96/None_Lookup/LookupTableFindV2.string_lookup_96/None_Lookup/LookupTableFindV22@
string_lookup_97/Assert/Assertstring_lookup_97/Assert/Assert2`
.string_lookup_97/None_Lookup/LookupTableFindV2.string_lookup_97/None_Lookup/LookupTableFindV22@
string_lookup_98/Assert/Assertstring_lookup_98/Assert/Assert2`
.string_lookup_98/None_Lookup/LookupTableFindV2.string_lookup_98/None_Lookup/LookupTableFindV22@
string_lookup_99/Assert/Assertstring_lookup_99/Assert/Assert2`
.string_lookup_99/None_Lookup/LookupTableFindV2.string_lookup_99/None_Lookup/LookupTableFindV22P
&user_embedding/StatefulPartitionedCall&user_embedding/StatefulPartitionedCall2r
7user_embedding/kernel/Regularizer/L2Loss/ReadVariableOp7user_embedding/kernel/Regularizer/L2Loss/ReadVariableOp:L H
'
_output_shapes
:€€€€€€€€€

_user_specified_nameBMI:RN
'
_output_shapes
:€€€€€€€€€
#
_user_specified_name	age_range:RN
'
_output_shapes
:€€€€€€€€€
#
_user_specified_name	allergens:PL
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	allergy:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
calories:VR
'
_output_shapes
:€€€€€€€€€
'
_user_specified_namecarbohydrates:XT
'
_output_shapes
:€€€€€€€€€
)
_user_specified_nameclinical_gender:XT
'
_output_shapes
:€€€€€€€€€
)
_user_specified_namecultural_factor:]Y
'
_output_shapes
:€€€€€€€€€
.
_user_specified_namecultural_restriction:_	[
'
_output_shapes
:€€€€€€€€€
0
_user_specified_namecurrent_daily_calories:_
[
'
_output_shapes
:€€€€€€€€€
0
_user_specified_namecurrent_working_status:SO
'
_output_shapes
:€€€€€€€€€
$
_user_specified_name
day_number:TP
(
_output_shapes
:€€€€€€€€€А
$
_user_specified_name
embeddings:RN
'
_output_shapes
:€€€€€€€€€
#
_user_specified_name	ethnicity:LH
'
_output_shapes
:€€€€€€€€€

_user_specified_namefat:NJ
'
_output_shapes
:€€€€€€€€€

_user_specified_namefiber:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameheight:SO
'
_output_shapes
:€€€€€€€€€
$
_user_specified_name
life_style:WS
'
_output_shapes
:€€€€€€€€€
(
_user_specified_namemarital_status:TP
'
_output_shapes
:€€€€€€€€€
%
_user_specified_namemeal_type_y:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
next_BMI:WS
'
_output_shapes
:€€€€€€€€€
(
_user_specified_namenutrition_goal:b^
'
_output_shapes
:€€€€€€€€€
3
_user_specified_nameplace_of_meal_consumption:NJ
'
_output_shapes
:€€€€€€€€€

_user_specified_nameprice:a]
'
_output_shapes
:€€€€€€€€€
2
_user_specified_nameprojected_daily_calories:PL
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	protein:mi
'
_output_shapes
:€€€€€€€€€
>
_user_specified_name&$social_situation_of_meal_consumption:NJ
'
_output_shapes
:€€€€€€€€€

_user_specified_nametaste:a]
'
_output_shapes
:€€€€€€€€€
2
_user_specified_nametime_of_meal_consumption:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameweight:,(
&
_user_specified_nametable_handle:

_output_shapes
: :, (
&
_user_specified_nametable_handle:!

_output_shapes
: :,"(
&
_user_specified_nametable_handle:#

_output_shapes
: :,$(
&
_user_specified_nametable_handle:%

_output_shapes
: :,&(
&
_user_specified_nametable_handle:'

_output_shapes
: :,((
&
_user_specified_nametable_handle:)

_output_shapes
: :,*(
&
_user_specified_nametable_handle:+

_output_shapes
: :,,(
&
_user_specified_nametable_handle:-

_output_shapes
: :,.(
&
_user_specified_nametable_handle:/

_output_shapes
: :,0(
&
_user_specified_nametable_handle:1

_output_shapes
: :,2(
&
_user_specified_nametable_handle:3

_output_shapes
: :,4(
&
_user_specified_nametable_handle:5

_output_shapes
: :,6(
&
_user_specified_nametable_handle:7

_output_shapes
: :,8(
&
_user_specified_nametable_handle:9

_output_shapes
: :,:(
&
_user_specified_nametable_handle:;

_output_shapes
: :'<#
!
_user_specified_name	7095106:'=#
!
_user_specified_name	7095117:'>#
!
_user_specified_name	7095128:'?#
!
_user_specified_name	7095139:'@#
!
_user_specified_name	7095150:'A#
!
_user_specified_name	7095161:'B#
!
_user_specified_name	7095172:'C#
!
_user_specified_name	7095183:'D#
!
_user_specified_name	7095194:'E#
!
_user_specified_name	7095205:'F#
!
_user_specified_name	7095216:'G#
!
_user_specified_name	7095227:'H#
!
_user_specified_name	7095238:'I#
!
_user_specified_name	7095249:'J#
!
_user_specified_name	7095260:,K(
&
_user_specified_nametable_handle:L

_output_shapes
: :,M(
&
_user_specified_nametable_handle:N

_output_shapes
: :'O#
!
_user_specified_name	7095473:'P#
!
_user_specified_name	7095475:'Q#
!
_user_specified_name	7095492:'R#
!
_user_specified_name	7095494:'S#
!
_user_specified_name	7095522:'T#
!
_user_specified_name	7095524:'U#
!
_user_specified_name	7095564:'V#
!
_user_specified_name	7095566:'W#
!
_user_specified_name	7095568:'X#
!
_user_specified_name	7095570:'Y#
!
_user_specified_name	7095588:'Z#
!
_user_specified_name	7095590:'[#
!
_user_specified_name	7095608:'\#
!
_user_specified_name	7095610:']#
!
_user_specified_name	7095628:'^#
!
_user_specified_name	7095630:'_#
!
_user_specified_name	7095648:'`#
!
_user_specified_name	7095650
°
І
I__inference_embedding_90_layer_call_and_return_conditional_losses_7095105

inputs	*
embedding_lookup_7095100:W	
identityИҐembedding_lookupї
embedding_lookupResourceGatherembedding_lookup_7095100inputs*
Tindices0	*+
_class!
loc:@embedding_lookup/7095100*+
_output_shapes
:€€€€€€€€€	*
dtype0v
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_output_shapes
:€€€€€€€€€	u
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€	5
NoOpNoOp^embedding_lookup*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:€€€€€€€€€: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:'#
!
_user_specified_name	7095100
Ь
.
__inference__destroyer_7098065
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
њ
c
G__inference_flatten_81_layer_call_and_return_conditional_losses_7095344

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:€€€€€€€€€X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
њ
c
G__inference_flatten_88_layer_call_and_return_conditional_losses_7097591

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:€€€€€€€€€X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ж
ґ
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_7094860

inputs0
!batchnorm_readvariableop_resource:	й4
%batchnorm_mul_readvariableop_resource:	й2
#batchnorm_readvariableop_1_resource:	й2
#batchnorm_readvariableop_2_resource:	й
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/ReadVariableOp_1Ґbatchnorm/ReadVariableOp_2Ґbatchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:й*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:йQ
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:й
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:й*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:йd
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:€€€€€€€€€й{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:й*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:й{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:й*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:йs
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:€€€€€€€€€йc
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€йЦ
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€й: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€й
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Х
p
$__inference__update_step_xla_7096976
gradient

gradient_1	

gradient_2
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:€€€€€€€€€:€€€€€€€€€:: *
	_noinline(:Q M
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
gradient:MI
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
gradient:D@

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
«
є
K__inference_food_embedding_layer_call_and_return_conditional_losses_7097637

inputs2
matmul_readvariableop_resource:
£ђ.
biasadd_readvariableop_resource:	ђ
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐ7food_embedding/kernel/Regularizer/L2Loss/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
£ђ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђШ
7food_embedding/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
£ђ*
dtype0Ф
(food_embedding/kernel/Regularizer/L2LossL2Loss?food_embedding/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: l
'food_embedding/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„£<≤
%food_embedding/kernel/Regularizer/mulMul0food_embedding/kernel/Regularizer/mul/x:output:01food_embedding/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: `
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€ђН
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp8^food_embedding/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€£: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2r
7food_embedding/kernel/Regularizer/L2Loss/ReadVariableOp7food_embedding/kernel/Regularizer/L2Loss/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€£
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
°
І
I__inference_embedding_88_layer_call_and_return_conditional_losses_7095248

inputs	*
embedding_lookup_7095243:
identityИҐembedding_lookupї
embedding_lookupResourceGatherembedding_lookup_7095243inputs*
Tindices0	*+
_class!
loc:@embedding_lookup/7095243*+
_output_shapes
:€€€€€€€€€*
dtype0v
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_output_shapes
:€€€€€€€€€u
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€5
NoOpNoOp^embedding_lookup*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:€€€€€€€€€: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:'#
!
_user_specified_name	7095243
°
І
I__inference_embedding_84_layer_call_and_return_conditional_losses_7095149

inputs	*
embedding_lookup_7095144:
identityИҐembedding_lookupї
embedding_lookupResourceGatherembedding_lookup_7095144inputs*
Tindices0	*+
_class!
loc:@embedding_lookup/7095144*+
_output_shapes
:€€€€€€€€€*
dtype0v
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_output_shapes
:€€€€€€€€€u
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€5
NoOpNoOp^embedding_lookup*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:€€€€€€€€€: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:'#
!
_user_specified_name	7095144
і
В
.__inference_embedding_81_layer_call_fn_7097215

inputs	
unknown:
identityИҐStatefulPartitionedCallЎ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_embedding_81_layer_call_and_return_conditional_losses_7095182s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:€€€€€€€€€: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:'#
!
_user_specified_name	7097211
ѓ
Е
 __inference__initializer_7098121:
6key_value_init6434863_lookuptableimportv2_table_handle2
.key_value_init6434863_lookuptableimportv2_keys4
0key_value_init6434863_lookuptableimportv2_values	
identityИҐ)key_value_init6434863/LookupTableImportV2З
)key_value_init6434863/LookupTableImportV2LookupTableImportV26key_value_init6434863_lookuptableimportv2_table_handle.key_value_init6434863_lookuptableimportv2_keys0key_value_init6434863_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: N
NoOpNoOp*^key_value_init6434863/LookupTableImportV2*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2V
)key_value_init6434863/LookupTableImportV2)key_value_init6434863/LookupTableImportV2:, (
&
_user_specified_nametable_handle: 

_output_shapes
:: 

_output_shapes
:
љ
з
K__inference_concatenate_33_layer_call_and_return_conditional_losses_7097569
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :»
concatConcatV2inputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9concat/axis:output:0*
N
*
T0*(
_output_shapes
:€€€€€€€€€£X
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:€€€€€€€€€£"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*‘
_input_shapes¬
њ:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€	:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€А:Q M
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_1:QM
'
_output_shapes
:€€€€€€€€€	
"
_user_specified_name
inputs_2:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_3:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_4:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_5:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_6:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_7:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_8:R	N
(
_output_shapes
:€€€€€€€€€А
"
_user_specified_name
inputs_9
ѓ
<
__inference__creator_7098024
identityИҐ
hash_tableo

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	6434558*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: /
NoOpNoOp^hash_table*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
њ
c
G__inference_flatten_89_layer_call_and_return_conditional_losses_7097460

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:€€€€€€€€€X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ѓ
<
__inference__creator_7098144
identityИҐ
hash_tableo

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	6435190*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: /
NoOpNoOp^hash_table*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
њ
c
G__inference_flatten_83_layer_call_and_return_conditional_losses_7095358

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:€€€€€€€€€X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
А
Ь
,__inference_fc_layer_0_layer_call_fn_7097817

inputs
unknown:
йА
	unknown_0:	А
identityИҐStatefulPartitionedCallа
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_fc_layer_0_layer_call_and_return_conditional_losses_7095587p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€А<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€й: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€й
 
_user_specified_nameinputs:'#
!
_user_specified_name	7097811:'#
!
_user_specified_name	7097813
ЖҐ
мd
 __inference__traced_save_7098981
file_prefix@
.read_disablecopyonread_embedding_76_embeddings:B
0read_1_disablecopyonread_embedding_77_embeddings:B
0read_2_disablecopyonread_embedding_78_embeddings:B
0read_3_disablecopyonread_embedding_79_embeddings:B
0read_4_disablecopyonread_embedding_80_embeddings:B
0read_5_disablecopyonread_embedding_81_embeddings:B
0read_6_disablecopyonread_embedding_82_embeddings:B
0read_7_disablecopyonread_embedding_83_embeddings:B
0read_8_disablecopyonread_embedding_84_embeddings:B
0read_9_disablecopyonread_embedding_85_embeddings:C
1read_10_disablecopyonread_embedding_86_embeddings:C
1read_11_disablecopyonread_embedding_89_embeddings:C
1read_12_disablecopyonread_embedding_90_embeddings:W	C
1read_13_disablecopyonread_embedding_87_embeddings:C
1read_14_disablecopyonread_embedding_88_embeddings:B
/read_15_disablecopyonread_user_embedding_kernel:	ђ<
-read_16_disablecopyonread_user_embedding_bias:	ђC
/read_17_disablecopyonread_food_embedding_kernel:
£ђ<
-read_18_disablecopyonread_food_embedding_bias:	ђE
2read_19_disablecopyonread_context_embedding_kernel:	Ш>
0read_20_disablecopyonread_context_embedding_bias:E
6read_21_disablecopyonread_batch_normalization_12_gamma:	йD
5read_22_disablecopyonread_batch_normalization_12_beta:	йK
<read_23_disablecopyonread_batch_normalization_12_moving_mean:	йO
@read_24_disablecopyonread_batch_normalization_12_moving_variance:	й?
+read_25_disablecopyonread_fc_layer_0_kernel:
йА8
)read_26_disablecopyonread_fc_layer_0_bias:	А>
+read_27_disablecopyonread_fc_layer_1_kernel:	А@7
)read_28_disablecopyonread_fc_layer_1_bias:@=
+read_29_disablecopyonread_fc_layer_2_kernel:@ 7
)read_30_disablecopyonread_fc_layer_2_bias: ;
)read_31_disablecopyonread_output_0_kernel: 5
'read_32_disablecopyonread_output_0_bias:-
#read_33_disablecopyonread_iteration:	 1
'read_34_disablecopyonread_learning_rate: J
8read_35_disablecopyonread_adam_m_embedding_76_embeddings:J
8read_36_disablecopyonread_adam_v_embedding_76_embeddings:J
8read_37_disablecopyonread_adam_m_embedding_77_embeddings:J
8read_38_disablecopyonread_adam_v_embedding_77_embeddings:J
8read_39_disablecopyonread_adam_m_embedding_78_embeddings:J
8read_40_disablecopyonread_adam_v_embedding_78_embeddings:J
8read_41_disablecopyonread_adam_m_embedding_79_embeddings:J
8read_42_disablecopyonread_adam_v_embedding_79_embeddings:J
8read_43_disablecopyonread_adam_m_embedding_80_embeddings:J
8read_44_disablecopyonread_adam_v_embedding_80_embeddings:J
8read_45_disablecopyonread_adam_m_embedding_81_embeddings:J
8read_46_disablecopyonread_adam_v_embedding_81_embeddings:J
8read_47_disablecopyonread_adam_m_embedding_82_embeddings:J
8read_48_disablecopyonread_adam_v_embedding_82_embeddings:J
8read_49_disablecopyonread_adam_m_embedding_83_embeddings:J
8read_50_disablecopyonread_adam_v_embedding_83_embeddings:J
8read_51_disablecopyonread_adam_m_embedding_84_embeddings:J
8read_52_disablecopyonread_adam_v_embedding_84_embeddings:J
8read_53_disablecopyonread_adam_m_embedding_85_embeddings:J
8read_54_disablecopyonread_adam_v_embedding_85_embeddings:J
8read_55_disablecopyonread_adam_m_embedding_86_embeddings:J
8read_56_disablecopyonread_adam_v_embedding_86_embeddings:J
8read_57_disablecopyonread_adam_m_embedding_89_embeddings:J
8read_58_disablecopyonread_adam_v_embedding_89_embeddings:J
8read_59_disablecopyonread_adam_m_embedding_90_embeddings:W	J
8read_60_disablecopyonread_adam_v_embedding_90_embeddings:W	J
8read_61_disablecopyonread_adam_m_embedding_87_embeddings:J
8read_62_disablecopyonread_adam_v_embedding_87_embeddings:J
8read_63_disablecopyonread_adam_m_embedding_88_embeddings:J
8read_64_disablecopyonread_adam_v_embedding_88_embeddings:I
6read_65_disablecopyonread_adam_m_user_embedding_kernel:	ђI
6read_66_disablecopyonread_adam_v_user_embedding_kernel:	ђC
4read_67_disablecopyonread_adam_m_user_embedding_bias:	ђC
4read_68_disablecopyonread_adam_v_user_embedding_bias:	ђJ
6read_69_disablecopyonread_adam_m_food_embedding_kernel:
£ђJ
6read_70_disablecopyonread_adam_v_food_embedding_kernel:
£ђC
4read_71_disablecopyonread_adam_m_food_embedding_bias:	ђC
4read_72_disablecopyonread_adam_v_food_embedding_bias:	ђL
9read_73_disablecopyonread_adam_m_context_embedding_kernel:	ШL
9read_74_disablecopyonread_adam_v_context_embedding_kernel:	ШE
7read_75_disablecopyonread_adam_m_context_embedding_bias:E
7read_76_disablecopyonread_adam_v_context_embedding_bias:L
=read_77_disablecopyonread_adam_m_batch_normalization_12_gamma:	йL
=read_78_disablecopyonread_adam_v_batch_normalization_12_gamma:	йK
<read_79_disablecopyonread_adam_m_batch_normalization_12_beta:	йK
<read_80_disablecopyonread_adam_v_batch_normalization_12_beta:	йF
2read_81_disablecopyonread_adam_m_fc_layer_0_kernel:
йАF
2read_82_disablecopyonread_adam_v_fc_layer_0_kernel:
йА?
0read_83_disablecopyonread_adam_m_fc_layer_0_bias:	А?
0read_84_disablecopyonread_adam_v_fc_layer_0_bias:	АE
2read_85_disablecopyonread_adam_m_fc_layer_1_kernel:	А@E
2read_86_disablecopyonread_adam_v_fc_layer_1_kernel:	А@>
0read_87_disablecopyonread_adam_m_fc_layer_1_bias:@>
0read_88_disablecopyonread_adam_v_fc_layer_1_bias:@D
2read_89_disablecopyonread_adam_m_fc_layer_2_kernel:@ D
2read_90_disablecopyonread_adam_v_fc_layer_2_kernel:@ >
0read_91_disablecopyonread_adam_m_fc_layer_2_bias: >
0read_92_disablecopyonread_adam_v_fc_layer_2_bias: B
0read_93_disablecopyonread_adam_m_output_0_kernel: B
0read_94_disablecopyonread_adam_v_output_0_kernel: <
.read_95_disablecopyonread_adam_m_output_0_bias:<
.read_96_disablecopyonread_adam_v_output_0_bias:+
!read_97_disablecopyonread_total_1: +
!read_98_disablecopyonread_count_1: )
read_99_disablecopyonread_total: *
 read_100_disablecopyonread_count: 9
+read_101_disablecopyonread_true_positives_1:8
*read_102_disablecopyonread_false_positives:7
)read_103_disablecopyonread_true_positives:8
*read_104_disablecopyonread_false_negatives:
savev2_const_51
identity_211ИҐMergeV2CheckpointsҐRead/DisableCopyOnReadҐRead/ReadVariableOpҐRead_1/DisableCopyOnReadҐRead_1/ReadVariableOpҐRead_10/DisableCopyOnReadҐRead_10/ReadVariableOpҐRead_100/DisableCopyOnReadҐRead_100/ReadVariableOpҐRead_101/DisableCopyOnReadҐRead_101/ReadVariableOpҐRead_102/DisableCopyOnReadҐRead_102/ReadVariableOpҐRead_103/DisableCopyOnReadҐRead_103/ReadVariableOpҐRead_104/DisableCopyOnReadҐRead_104/ReadVariableOpҐRead_11/DisableCopyOnReadҐRead_11/ReadVariableOpҐRead_12/DisableCopyOnReadҐRead_12/ReadVariableOpҐRead_13/DisableCopyOnReadҐRead_13/ReadVariableOpҐRead_14/DisableCopyOnReadҐRead_14/ReadVariableOpҐRead_15/DisableCopyOnReadҐRead_15/ReadVariableOpҐRead_16/DisableCopyOnReadҐRead_16/ReadVariableOpҐRead_17/DisableCopyOnReadҐRead_17/ReadVariableOpҐRead_18/DisableCopyOnReadҐRead_18/ReadVariableOpҐRead_19/DisableCopyOnReadҐRead_19/ReadVariableOpҐRead_2/DisableCopyOnReadҐRead_2/ReadVariableOpҐRead_20/DisableCopyOnReadҐRead_20/ReadVariableOpҐRead_21/DisableCopyOnReadҐRead_21/ReadVariableOpҐRead_22/DisableCopyOnReadҐRead_22/ReadVariableOpҐRead_23/DisableCopyOnReadҐRead_23/ReadVariableOpҐRead_24/DisableCopyOnReadҐRead_24/ReadVariableOpҐRead_25/DisableCopyOnReadҐRead_25/ReadVariableOpҐRead_26/DisableCopyOnReadҐRead_26/ReadVariableOpҐRead_27/DisableCopyOnReadҐRead_27/ReadVariableOpҐRead_28/DisableCopyOnReadҐRead_28/ReadVariableOpҐRead_29/DisableCopyOnReadҐRead_29/ReadVariableOpҐRead_3/DisableCopyOnReadҐRead_3/ReadVariableOpҐRead_30/DisableCopyOnReadҐRead_30/ReadVariableOpҐRead_31/DisableCopyOnReadҐRead_31/ReadVariableOpҐRead_32/DisableCopyOnReadҐRead_32/ReadVariableOpҐRead_33/DisableCopyOnReadҐRead_33/ReadVariableOpҐRead_34/DisableCopyOnReadҐRead_34/ReadVariableOpҐRead_35/DisableCopyOnReadҐRead_35/ReadVariableOpҐRead_36/DisableCopyOnReadҐRead_36/ReadVariableOpҐRead_37/DisableCopyOnReadҐRead_37/ReadVariableOpҐRead_38/DisableCopyOnReadҐRead_38/ReadVariableOpҐRead_39/DisableCopyOnReadҐRead_39/ReadVariableOpҐRead_4/DisableCopyOnReadҐRead_4/ReadVariableOpҐRead_40/DisableCopyOnReadҐRead_40/ReadVariableOpҐRead_41/DisableCopyOnReadҐRead_41/ReadVariableOpҐRead_42/DisableCopyOnReadҐRead_42/ReadVariableOpҐRead_43/DisableCopyOnReadҐRead_43/ReadVariableOpҐRead_44/DisableCopyOnReadҐRead_44/ReadVariableOpҐRead_45/DisableCopyOnReadҐRead_45/ReadVariableOpҐRead_46/DisableCopyOnReadҐRead_46/ReadVariableOpҐRead_47/DisableCopyOnReadҐRead_47/ReadVariableOpҐRead_48/DisableCopyOnReadҐRead_48/ReadVariableOpҐRead_49/DisableCopyOnReadҐRead_49/ReadVariableOpҐRead_5/DisableCopyOnReadҐRead_5/ReadVariableOpҐRead_50/DisableCopyOnReadҐRead_50/ReadVariableOpҐRead_51/DisableCopyOnReadҐRead_51/ReadVariableOpҐRead_52/DisableCopyOnReadҐRead_52/ReadVariableOpҐRead_53/DisableCopyOnReadҐRead_53/ReadVariableOpҐRead_54/DisableCopyOnReadҐRead_54/ReadVariableOpҐRead_55/DisableCopyOnReadҐRead_55/ReadVariableOpҐRead_56/DisableCopyOnReadҐRead_56/ReadVariableOpҐRead_57/DisableCopyOnReadҐRead_57/ReadVariableOpҐRead_58/DisableCopyOnReadҐRead_58/ReadVariableOpҐRead_59/DisableCopyOnReadҐRead_59/ReadVariableOpҐRead_6/DisableCopyOnReadҐRead_6/ReadVariableOpҐRead_60/DisableCopyOnReadҐRead_60/ReadVariableOpҐRead_61/DisableCopyOnReadҐRead_61/ReadVariableOpҐRead_62/DisableCopyOnReadҐRead_62/ReadVariableOpҐRead_63/DisableCopyOnReadҐRead_63/ReadVariableOpҐRead_64/DisableCopyOnReadҐRead_64/ReadVariableOpҐRead_65/DisableCopyOnReadҐRead_65/ReadVariableOpҐRead_66/DisableCopyOnReadҐRead_66/ReadVariableOpҐRead_67/DisableCopyOnReadҐRead_67/ReadVariableOpҐRead_68/DisableCopyOnReadҐRead_68/ReadVariableOpҐRead_69/DisableCopyOnReadҐRead_69/ReadVariableOpҐRead_7/DisableCopyOnReadҐRead_7/ReadVariableOpҐRead_70/DisableCopyOnReadҐRead_70/ReadVariableOpҐRead_71/DisableCopyOnReadҐRead_71/ReadVariableOpҐRead_72/DisableCopyOnReadҐRead_72/ReadVariableOpҐRead_73/DisableCopyOnReadҐRead_73/ReadVariableOpҐRead_74/DisableCopyOnReadҐRead_74/ReadVariableOpҐRead_75/DisableCopyOnReadҐRead_75/ReadVariableOpҐRead_76/DisableCopyOnReadҐRead_76/ReadVariableOpҐRead_77/DisableCopyOnReadҐRead_77/ReadVariableOpҐRead_78/DisableCopyOnReadҐRead_78/ReadVariableOpҐRead_79/DisableCopyOnReadҐRead_79/ReadVariableOpҐRead_8/DisableCopyOnReadҐRead_8/ReadVariableOpҐRead_80/DisableCopyOnReadҐRead_80/ReadVariableOpҐRead_81/DisableCopyOnReadҐRead_81/ReadVariableOpҐRead_82/DisableCopyOnReadҐRead_82/ReadVariableOpҐRead_83/DisableCopyOnReadҐRead_83/ReadVariableOpҐRead_84/DisableCopyOnReadҐRead_84/ReadVariableOpҐRead_85/DisableCopyOnReadҐRead_85/ReadVariableOpҐRead_86/DisableCopyOnReadҐRead_86/ReadVariableOpҐRead_87/DisableCopyOnReadҐRead_87/ReadVariableOpҐRead_88/DisableCopyOnReadҐRead_88/ReadVariableOpҐRead_89/DisableCopyOnReadҐRead_89/ReadVariableOpҐRead_9/DisableCopyOnReadҐRead_9/ReadVariableOpҐRead_90/DisableCopyOnReadҐRead_90/ReadVariableOpҐRead_91/DisableCopyOnReadҐRead_91/ReadVariableOpҐRead_92/DisableCopyOnReadҐRead_92/ReadVariableOpҐRead_93/DisableCopyOnReadҐRead_93/ReadVariableOpҐRead_94/DisableCopyOnReadҐRead_94/ReadVariableOpҐRead_95/DisableCopyOnReadҐRead_95/ReadVariableOpҐRead_96/DisableCopyOnReadҐRead_96/ReadVariableOpҐRead_97/DisableCopyOnReadҐRead_97/ReadVariableOpҐRead_98/DisableCopyOnReadҐRead_98/ReadVariableOpҐRead_99/DisableCopyOnReadҐRead_99/ReadVariableOpw
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
_temp/partБ
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
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: А
Read/DisableCopyOnReadDisableCopyOnRead.read_disablecopyonread_embedding_76_embeddings"/device:CPU:0*
_output_shapes
 ™
Read/ReadVariableOpReadVariableOp.read_disablecopyonread_embedding_76_embeddings^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0i
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:a

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes

:Д
Read_1/DisableCopyOnReadDisableCopyOnRead0read_1_disablecopyonread_embedding_77_embeddings"/device:CPU:0*
_output_shapes
 ∞
Read_1/ReadVariableOpReadVariableOp0read_1_disablecopyonread_embedding_77_embeddings^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0m

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:c

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes

:Д
Read_2/DisableCopyOnReadDisableCopyOnRead0read_2_disablecopyonread_embedding_78_embeddings"/device:CPU:0*
_output_shapes
 ∞
Read_2/ReadVariableOpReadVariableOp0read_2_disablecopyonread_embedding_78_embeddings^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0m

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:c

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes

:Д
Read_3/DisableCopyOnReadDisableCopyOnRead0read_3_disablecopyonread_embedding_79_embeddings"/device:CPU:0*
_output_shapes
 ∞
Read_3/ReadVariableOpReadVariableOp0read_3_disablecopyonread_embedding_79_embeddings^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0m

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:c

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes

:Д
Read_4/DisableCopyOnReadDisableCopyOnRead0read_4_disablecopyonread_embedding_80_embeddings"/device:CPU:0*
_output_shapes
 ∞
Read_4/ReadVariableOpReadVariableOp0read_4_disablecopyonread_embedding_80_embeddings^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0m

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:c

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes

:Д
Read_5/DisableCopyOnReadDisableCopyOnRead0read_5_disablecopyonread_embedding_81_embeddings"/device:CPU:0*
_output_shapes
 ∞
Read_5/ReadVariableOpReadVariableOp0read_5_disablecopyonread_embedding_81_embeddings^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0n
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes

:Д
Read_6/DisableCopyOnReadDisableCopyOnRead0read_6_disablecopyonread_embedding_82_embeddings"/device:CPU:0*
_output_shapes
 ∞
Read_6/ReadVariableOpReadVariableOp0read_6_disablecopyonread_embedding_82_embeddings^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0n
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes

:Д
Read_7/DisableCopyOnReadDisableCopyOnRead0read_7_disablecopyonread_embedding_83_embeddings"/device:CPU:0*
_output_shapes
 ∞
Read_7/ReadVariableOpReadVariableOp0read_7_disablecopyonread_embedding_83_embeddings^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0n
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes

:Д
Read_8/DisableCopyOnReadDisableCopyOnRead0read_8_disablecopyonread_embedding_84_embeddings"/device:CPU:0*
_output_shapes
 ∞
Read_8/ReadVariableOpReadVariableOp0read_8_disablecopyonread_embedding_84_embeddings^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0n
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes

:Д
Read_9/DisableCopyOnReadDisableCopyOnRead0read_9_disablecopyonread_embedding_85_embeddings"/device:CPU:0*
_output_shapes
 ∞
Read_9/ReadVariableOpReadVariableOp0read_9_disablecopyonread_embedding_85_embeddings^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0n
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes

:Ж
Read_10/DisableCopyOnReadDisableCopyOnRead1read_10_disablecopyonread_embedding_86_embeddings"/device:CPU:0*
_output_shapes
 ≥
Read_10/ReadVariableOpReadVariableOp1read_10_disablecopyonread_embedding_86_embeddings^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes

:Ж
Read_11/DisableCopyOnReadDisableCopyOnRead1read_11_disablecopyonread_embedding_89_embeddings"/device:CPU:0*
_output_shapes
 ≥
Read_11/ReadVariableOpReadVariableOp1read_11_disablecopyonread_embedding_89_embeddings^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes

:Ж
Read_12/DisableCopyOnReadDisableCopyOnRead1read_12_disablecopyonread_embedding_90_embeddings"/device:CPU:0*
_output_shapes
 ≥
Read_12/ReadVariableOpReadVariableOp1read_12_disablecopyonread_embedding_90_embeddings^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:W	*
dtype0o
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:W	e
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes

:W	Ж
Read_13/DisableCopyOnReadDisableCopyOnRead1read_13_disablecopyonread_embedding_87_embeddings"/device:CPU:0*
_output_shapes
 ≥
Read_13/ReadVariableOpReadVariableOp1read_13_disablecopyonread_embedding_87_embeddings^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes

:Ж
Read_14/DisableCopyOnReadDisableCopyOnRead1read_14_disablecopyonread_embedding_88_embeddings"/device:CPU:0*
_output_shapes
 ≥
Read_14/ReadVariableOpReadVariableOp1read_14_disablecopyonread_embedding_88_embeddings^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes

:Д
Read_15/DisableCopyOnReadDisableCopyOnRead/read_15_disablecopyonread_user_embedding_kernel"/device:CPU:0*
_output_shapes
 ≤
Read_15/ReadVariableOpReadVariableOp/read_15_disablecopyonread_user_embedding_kernel^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	ђ*
dtype0p
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	ђf
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
:	ђВ
Read_16/DisableCopyOnReadDisableCopyOnRead-read_16_disablecopyonread_user_embedding_bias"/device:CPU:0*
_output_shapes
 ђ
Read_16/ReadVariableOpReadVariableOp-read_16_disablecopyonread_user_embedding_bias^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:ђ*
dtype0l
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:ђb
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes	
:ђД
Read_17/DisableCopyOnReadDisableCopyOnRead/read_17_disablecopyonread_food_embedding_kernel"/device:CPU:0*
_output_shapes
 ≥
Read_17/ReadVariableOpReadVariableOp/read_17_disablecopyonread_food_embedding_kernel^Read_17/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
£ђ*
dtype0q
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
£ђg
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0* 
_output_shapes
:
£ђВ
Read_18/DisableCopyOnReadDisableCopyOnRead-read_18_disablecopyonread_food_embedding_bias"/device:CPU:0*
_output_shapes
 ђ
Read_18/ReadVariableOpReadVariableOp-read_18_disablecopyonread_food_embedding_bias^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:ђ*
dtype0l
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:ђb
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes	
:ђЗ
Read_19/DisableCopyOnReadDisableCopyOnRead2read_19_disablecopyonread_context_embedding_kernel"/device:CPU:0*
_output_shapes
 µ
Read_19/ReadVariableOpReadVariableOp2read_19_disablecopyonread_context_embedding_kernel^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	Ш*
dtype0p
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	Шf
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
:	ШЕ
Read_20/DisableCopyOnReadDisableCopyOnRead0read_20_disablecopyonread_context_embedding_bias"/device:CPU:0*
_output_shapes
 Ѓ
Read_20/ReadVariableOpReadVariableOp0read_20_disablecopyonread_context_embedding_bias^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes
:Л
Read_21/DisableCopyOnReadDisableCopyOnRead6read_21_disablecopyonread_batch_normalization_12_gamma"/device:CPU:0*
_output_shapes
 µ
Read_21/ReadVariableOpReadVariableOp6read_21_disablecopyonread_batch_normalization_12_gamma^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:й*
dtype0l
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:йb
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes	
:йК
Read_22/DisableCopyOnReadDisableCopyOnRead5read_22_disablecopyonread_batch_normalization_12_beta"/device:CPU:0*
_output_shapes
 і
Read_22/ReadVariableOpReadVariableOp5read_22_disablecopyonread_batch_normalization_12_beta^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:й*
dtype0l
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:йb
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes	
:йС
Read_23/DisableCopyOnReadDisableCopyOnRead<read_23_disablecopyonread_batch_normalization_12_moving_mean"/device:CPU:0*
_output_shapes
 ї
Read_23/ReadVariableOpReadVariableOp<read_23_disablecopyonread_batch_normalization_12_moving_mean^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:й*
dtype0l
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:йb
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes	
:йХ
Read_24/DisableCopyOnReadDisableCopyOnRead@read_24_disablecopyonread_batch_normalization_12_moving_variance"/device:CPU:0*
_output_shapes
 њ
Read_24/ReadVariableOpReadVariableOp@read_24_disablecopyonread_batch_normalization_12_moving_variance^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:й*
dtype0l
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:йb
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes	
:йА
Read_25/DisableCopyOnReadDisableCopyOnRead+read_25_disablecopyonread_fc_layer_0_kernel"/device:CPU:0*
_output_shapes
 ѓ
Read_25/ReadVariableOpReadVariableOp+read_25_disablecopyonread_fc_layer_0_kernel^Read_25/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
йА*
dtype0q
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
йАg
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0* 
_output_shapes
:
йА~
Read_26/DisableCopyOnReadDisableCopyOnRead)read_26_disablecopyonread_fc_layer_0_bias"/device:CPU:0*
_output_shapes
 ®
Read_26/ReadVariableOpReadVariableOp)read_26_disablecopyonread_fc_layer_0_bias^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0l
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аb
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes	
:АА
Read_27/DisableCopyOnReadDisableCopyOnRead+read_27_disablecopyonread_fc_layer_1_kernel"/device:CPU:0*
_output_shapes
 Ѓ
Read_27/ReadVariableOpReadVariableOp+read_27_disablecopyonread_fc_layer_1_kernel^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	А@*
dtype0p
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	А@f
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes
:	А@~
Read_28/DisableCopyOnReadDisableCopyOnRead)read_28_disablecopyonread_fc_layer_1_bias"/device:CPU:0*
_output_shapes
 І
Read_28/ReadVariableOpReadVariableOp)read_28_disablecopyonread_fc_layer_1_bias^Read_28/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes
:@А
Read_29/DisableCopyOnReadDisableCopyOnRead+read_29_disablecopyonread_fc_layer_2_kernel"/device:CPU:0*
_output_shapes
 ≠
Read_29/ReadVariableOpReadVariableOp+read_29_disablecopyonread_fc_layer_2_kernel^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@ *
dtype0o
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@ e
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes

:@ ~
Read_30/DisableCopyOnReadDisableCopyOnRead)read_30_disablecopyonread_fc_layer_2_bias"/device:CPU:0*
_output_shapes
 І
Read_30/ReadVariableOpReadVariableOp)read_30_disablecopyonread_fc_layer_2_bias^Read_30/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0k
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: a
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*
_output_shapes
: ~
Read_31/DisableCopyOnReadDisableCopyOnRead)read_31_disablecopyonread_output_0_kernel"/device:CPU:0*
_output_shapes
 Ђ
Read_31/ReadVariableOpReadVariableOp)read_31_disablecopyonread_output_0_kernel^Read_31/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0o
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: e
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes

: |
Read_32/DisableCopyOnReadDisableCopyOnRead'read_32_disablecopyonread_output_0_bias"/device:CPU:0*
_output_shapes
 •
Read_32/ReadVariableOpReadVariableOp'read_32_disablecopyonread_output_0_bias^Read_32/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*
_output_shapes
:x
Read_33/DisableCopyOnReadDisableCopyOnRead#read_33_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 Э
Read_33/ReadVariableOpReadVariableOp#read_33_disablecopyonread_iteration^Read_33/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0	*
_output_shapes
: |
Read_34/DisableCopyOnReadDisableCopyOnRead'read_34_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 °
Read_34/ReadVariableOpReadVariableOp'read_34_disablecopyonread_learning_rate^Read_34/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*
_output_shapes
: Н
Read_35/DisableCopyOnReadDisableCopyOnRead8read_35_disablecopyonread_adam_m_embedding_76_embeddings"/device:CPU:0*
_output_shapes
 Ї
Read_35/ReadVariableOpReadVariableOp8read_35_disablecopyonread_adam_m_embedding_76_embeddings^Read_35/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*
_output_shapes

:Н
Read_36/DisableCopyOnReadDisableCopyOnRead8read_36_disablecopyonread_adam_v_embedding_76_embeddings"/device:CPU:0*
_output_shapes
 Ї
Read_36/ReadVariableOpReadVariableOp8read_36_disablecopyonread_adam_v_embedding_76_embeddings^Read_36/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*
_output_shapes

:Н
Read_37/DisableCopyOnReadDisableCopyOnRead8read_37_disablecopyonread_adam_m_embedding_77_embeddings"/device:CPU:0*
_output_shapes
 Ї
Read_37/ReadVariableOpReadVariableOp8read_37_disablecopyonread_adam_m_embedding_77_embeddings^Read_37/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_74IdentityRead_37/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*
_output_shapes

:Н
Read_38/DisableCopyOnReadDisableCopyOnRead8read_38_disablecopyonread_adam_v_embedding_77_embeddings"/device:CPU:0*
_output_shapes
 Ї
Read_38/ReadVariableOpReadVariableOp8read_38_disablecopyonread_adam_v_embedding_77_embeddings^Read_38/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_76IdentityRead_38/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0*
_output_shapes

:Н
Read_39/DisableCopyOnReadDisableCopyOnRead8read_39_disablecopyonread_adam_m_embedding_78_embeddings"/device:CPU:0*
_output_shapes
 Ї
Read_39/ReadVariableOpReadVariableOp8read_39_disablecopyonread_adam_m_embedding_78_embeddings^Read_39/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_78IdentityRead_39/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0*
_output_shapes

:Н
Read_40/DisableCopyOnReadDisableCopyOnRead8read_40_disablecopyonread_adam_v_embedding_78_embeddings"/device:CPU:0*
_output_shapes
 Ї
Read_40/ReadVariableOpReadVariableOp8read_40_disablecopyonread_adam_v_embedding_78_embeddings^Read_40/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_80IdentityRead_40/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0*
_output_shapes

:Н
Read_41/DisableCopyOnReadDisableCopyOnRead8read_41_disablecopyonread_adam_m_embedding_79_embeddings"/device:CPU:0*
_output_shapes
 Ї
Read_41/ReadVariableOpReadVariableOp8read_41_disablecopyonread_adam_m_embedding_79_embeddings^Read_41/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_82IdentityRead_41/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*
_output_shapes

:Н
Read_42/DisableCopyOnReadDisableCopyOnRead8read_42_disablecopyonread_adam_v_embedding_79_embeddings"/device:CPU:0*
_output_shapes
 Ї
Read_42/ReadVariableOpReadVariableOp8read_42_disablecopyonread_adam_v_embedding_79_embeddings^Read_42/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_84IdentityRead_42/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_85IdentityIdentity_84:output:0"/device:CPU:0*
T0*
_output_shapes

:Н
Read_43/DisableCopyOnReadDisableCopyOnRead8read_43_disablecopyonread_adam_m_embedding_80_embeddings"/device:CPU:0*
_output_shapes
 Ї
Read_43/ReadVariableOpReadVariableOp8read_43_disablecopyonread_adam_m_embedding_80_embeddings^Read_43/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_86IdentityRead_43/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_87IdentityIdentity_86:output:0"/device:CPU:0*
T0*
_output_shapes

:Н
Read_44/DisableCopyOnReadDisableCopyOnRead8read_44_disablecopyonread_adam_v_embedding_80_embeddings"/device:CPU:0*
_output_shapes
 Ї
Read_44/ReadVariableOpReadVariableOp8read_44_disablecopyonread_adam_v_embedding_80_embeddings^Read_44/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_88IdentityRead_44/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_89IdentityIdentity_88:output:0"/device:CPU:0*
T0*
_output_shapes

:Н
Read_45/DisableCopyOnReadDisableCopyOnRead8read_45_disablecopyonread_adam_m_embedding_81_embeddings"/device:CPU:0*
_output_shapes
 Ї
Read_45/ReadVariableOpReadVariableOp8read_45_disablecopyonread_adam_m_embedding_81_embeddings^Read_45/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_90IdentityRead_45/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_91IdentityIdentity_90:output:0"/device:CPU:0*
T0*
_output_shapes

:Н
Read_46/DisableCopyOnReadDisableCopyOnRead8read_46_disablecopyonread_adam_v_embedding_81_embeddings"/device:CPU:0*
_output_shapes
 Ї
Read_46/ReadVariableOpReadVariableOp8read_46_disablecopyonread_adam_v_embedding_81_embeddings^Read_46/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_92IdentityRead_46/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_93IdentityIdentity_92:output:0"/device:CPU:0*
T0*
_output_shapes

:Н
Read_47/DisableCopyOnReadDisableCopyOnRead8read_47_disablecopyonread_adam_m_embedding_82_embeddings"/device:CPU:0*
_output_shapes
 Ї
Read_47/ReadVariableOpReadVariableOp8read_47_disablecopyonread_adam_m_embedding_82_embeddings^Read_47/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_94IdentityRead_47/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_95IdentityIdentity_94:output:0"/device:CPU:0*
T0*
_output_shapes

:Н
Read_48/DisableCopyOnReadDisableCopyOnRead8read_48_disablecopyonread_adam_v_embedding_82_embeddings"/device:CPU:0*
_output_shapes
 Ї
Read_48/ReadVariableOpReadVariableOp8read_48_disablecopyonread_adam_v_embedding_82_embeddings^Read_48/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_96IdentityRead_48/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_97IdentityIdentity_96:output:0"/device:CPU:0*
T0*
_output_shapes

:Н
Read_49/DisableCopyOnReadDisableCopyOnRead8read_49_disablecopyonread_adam_m_embedding_83_embeddings"/device:CPU:0*
_output_shapes
 Ї
Read_49/ReadVariableOpReadVariableOp8read_49_disablecopyonread_adam_m_embedding_83_embeddings^Read_49/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_98IdentityRead_49/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_99IdentityIdentity_98:output:0"/device:CPU:0*
T0*
_output_shapes

:Н
Read_50/DisableCopyOnReadDisableCopyOnRead8read_50_disablecopyonread_adam_v_embedding_83_embeddings"/device:CPU:0*
_output_shapes
 Ї
Read_50/ReadVariableOpReadVariableOp8read_50_disablecopyonread_adam_v_embedding_83_embeddings^Read_50/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0p
Identity_100IdentityRead_50/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:g
Identity_101IdentityIdentity_100:output:0"/device:CPU:0*
T0*
_output_shapes

:Н
Read_51/DisableCopyOnReadDisableCopyOnRead8read_51_disablecopyonread_adam_m_embedding_84_embeddings"/device:CPU:0*
_output_shapes
 Ї
Read_51/ReadVariableOpReadVariableOp8read_51_disablecopyonread_adam_m_embedding_84_embeddings^Read_51/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0p
Identity_102IdentityRead_51/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:g
Identity_103IdentityIdentity_102:output:0"/device:CPU:0*
T0*
_output_shapes

:Н
Read_52/DisableCopyOnReadDisableCopyOnRead8read_52_disablecopyonread_adam_v_embedding_84_embeddings"/device:CPU:0*
_output_shapes
 Ї
Read_52/ReadVariableOpReadVariableOp8read_52_disablecopyonread_adam_v_embedding_84_embeddings^Read_52/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0p
Identity_104IdentityRead_52/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:g
Identity_105IdentityIdentity_104:output:0"/device:CPU:0*
T0*
_output_shapes

:Н
Read_53/DisableCopyOnReadDisableCopyOnRead8read_53_disablecopyonread_adam_m_embedding_85_embeddings"/device:CPU:0*
_output_shapes
 Ї
Read_53/ReadVariableOpReadVariableOp8read_53_disablecopyonread_adam_m_embedding_85_embeddings^Read_53/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0p
Identity_106IdentityRead_53/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:g
Identity_107IdentityIdentity_106:output:0"/device:CPU:0*
T0*
_output_shapes

:Н
Read_54/DisableCopyOnReadDisableCopyOnRead8read_54_disablecopyonread_adam_v_embedding_85_embeddings"/device:CPU:0*
_output_shapes
 Ї
Read_54/ReadVariableOpReadVariableOp8read_54_disablecopyonread_adam_v_embedding_85_embeddings^Read_54/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0p
Identity_108IdentityRead_54/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:g
Identity_109IdentityIdentity_108:output:0"/device:CPU:0*
T0*
_output_shapes

:Н
Read_55/DisableCopyOnReadDisableCopyOnRead8read_55_disablecopyonread_adam_m_embedding_86_embeddings"/device:CPU:0*
_output_shapes
 Ї
Read_55/ReadVariableOpReadVariableOp8read_55_disablecopyonread_adam_m_embedding_86_embeddings^Read_55/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0p
Identity_110IdentityRead_55/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:g
Identity_111IdentityIdentity_110:output:0"/device:CPU:0*
T0*
_output_shapes

:Н
Read_56/DisableCopyOnReadDisableCopyOnRead8read_56_disablecopyonread_adam_v_embedding_86_embeddings"/device:CPU:0*
_output_shapes
 Ї
Read_56/ReadVariableOpReadVariableOp8read_56_disablecopyonread_adam_v_embedding_86_embeddings^Read_56/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0p
Identity_112IdentityRead_56/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:g
Identity_113IdentityIdentity_112:output:0"/device:CPU:0*
T0*
_output_shapes

:Н
Read_57/DisableCopyOnReadDisableCopyOnRead8read_57_disablecopyonread_adam_m_embedding_89_embeddings"/device:CPU:0*
_output_shapes
 Ї
Read_57/ReadVariableOpReadVariableOp8read_57_disablecopyonread_adam_m_embedding_89_embeddings^Read_57/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0p
Identity_114IdentityRead_57/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:g
Identity_115IdentityIdentity_114:output:0"/device:CPU:0*
T0*
_output_shapes

:Н
Read_58/DisableCopyOnReadDisableCopyOnRead8read_58_disablecopyonread_adam_v_embedding_89_embeddings"/device:CPU:0*
_output_shapes
 Ї
Read_58/ReadVariableOpReadVariableOp8read_58_disablecopyonread_adam_v_embedding_89_embeddings^Read_58/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0p
Identity_116IdentityRead_58/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:g
Identity_117IdentityIdentity_116:output:0"/device:CPU:0*
T0*
_output_shapes

:Н
Read_59/DisableCopyOnReadDisableCopyOnRead8read_59_disablecopyonread_adam_m_embedding_90_embeddings"/device:CPU:0*
_output_shapes
 Ї
Read_59/ReadVariableOpReadVariableOp8read_59_disablecopyonread_adam_m_embedding_90_embeddings^Read_59/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:W	*
dtype0p
Identity_118IdentityRead_59/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:W	g
Identity_119IdentityIdentity_118:output:0"/device:CPU:0*
T0*
_output_shapes

:W	Н
Read_60/DisableCopyOnReadDisableCopyOnRead8read_60_disablecopyonread_adam_v_embedding_90_embeddings"/device:CPU:0*
_output_shapes
 Ї
Read_60/ReadVariableOpReadVariableOp8read_60_disablecopyonread_adam_v_embedding_90_embeddings^Read_60/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:W	*
dtype0p
Identity_120IdentityRead_60/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:W	g
Identity_121IdentityIdentity_120:output:0"/device:CPU:0*
T0*
_output_shapes

:W	Н
Read_61/DisableCopyOnReadDisableCopyOnRead8read_61_disablecopyonread_adam_m_embedding_87_embeddings"/device:CPU:0*
_output_shapes
 Ї
Read_61/ReadVariableOpReadVariableOp8read_61_disablecopyonread_adam_m_embedding_87_embeddings^Read_61/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0p
Identity_122IdentityRead_61/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:g
Identity_123IdentityIdentity_122:output:0"/device:CPU:0*
T0*
_output_shapes

:Н
Read_62/DisableCopyOnReadDisableCopyOnRead8read_62_disablecopyonread_adam_v_embedding_87_embeddings"/device:CPU:0*
_output_shapes
 Ї
Read_62/ReadVariableOpReadVariableOp8read_62_disablecopyonread_adam_v_embedding_87_embeddings^Read_62/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0p
Identity_124IdentityRead_62/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:g
Identity_125IdentityIdentity_124:output:0"/device:CPU:0*
T0*
_output_shapes

:Н
Read_63/DisableCopyOnReadDisableCopyOnRead8read_63_disablecopyonread_adam_m_embedding_88_embeddings"/device:CPU:0*
_output_shapes
 Ї
Read_63/ReadVariableOpReadVariableOp8read_63_disablecopyonread_adam_m_embedding_88_embeddings^Read_63/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0p
Identity_126IdentityRead_63/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:g
Identity_127IdentityIdentity_126:output:0"/device:CPU:0*
T0*
_output_shapes

:Н
Read_64/DisableCopyOnReadDisableCopyOnRead8read_64_disablecopyonread_adam_v_embedding_88_embeddings"/device:CPU:0*
_output_shapes
 Ї
Read_64/ReadVariableOpReadVariableOp8read_64_disablecopyonread_adam_v_embedding_88_embeddings^Read_64/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0p
Identity_128IdentityRead_64/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:g
Identity_129IdentityIdentity_128:output:0"/device:CPU:0*
T0*
_output_shapes

:Л
Read_65/DisableCopyOnReadDisableCopyOnRead6read_65_disablecopyonread_adam_m_user_embedding_kernel"/device:CPU:0*
_output_shapes
 є
Read_65/ReadVariableOpReadVariableOp6read_65_disablecopyonread_adam_m_user_embedding_kernel^Read_65/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	ђ*
dtype0q
Identity_130IdentityRead_65/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	ђh
Identity_131IdentityIdentity_130:output:0"/device:CPU:0*
T0*
_output_shapes
:	ђЛ
Read_66/DisableCopyOnReadDisableCopyOnRead6read_66_disablecopyonread_adam_v_user_embedding_kernel"/device:CPU:0*
_output_shapes
 є
Read_66/ReadVariableOpReadVariableOp6read_66_disablecopyonread_adam_v_user_embedding_kernel^Read_66/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	ђ*
dtype0q
Identity_132IdentityRead_66/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	ђh
Identity_133IdentityIdentity_132:output:0"/device:CPU:0*
T0*
_output_shapes
:	ђЙ
Read_67/DisableCopyOnReadDisableCopyOnRead4read_67_disablecopyonread_adam_m_user_embedding_bias"/device:CPU:0*
_output_shapes
 ≥
Read_67/ReadVariableOpReadVariableOp4read_67_disablecopyonread_adam_m_user_embedding_bias^Read_67/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:ђ*
dtype0m
Identity_134IdentityRead_67/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:ђd
Identity_135IdentityIdentity_134:output:0"/device:CPU:0*
T0*
_output_shapes	
:ђЙ
Read_68/DisableCopyOnReadDisableCopyOnRead4read_68_disablecopyonread_adam_v_user_embedding_bias"/device:CPU:0*
_output_shapes
 ≥
Read_68/ReadVariableOpReadVariableOp4read_68_disablecopyonread_adam_v_user_embedding_bias^Read_68/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:ђ*
dtype0m
Identity_136IdentityRead_68/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:ђd
Identity_137IdentityIdentity_136:output:0"/device:CPU:0*
T0*
_output_shapes	
:ђЛ
Read_69/DisableCopyOnReadDisableCopyOnRead6read_69_disablecopyonread_adam_m_food_embedding_kernel"/device:CPU:0*
_output_shapes
 Ї
Read_69/ReadVariableOpReadVariableOp6read_69_disablecopyonread_adam_m_food_embedding_kernel^Read_69/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
£ђ*
dtype0r
Identity_138IdentityRead_69/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
£ђi
Identity_139IdentityIdentity_138:output:0"/device:CPU:0*
T0* 
_output_shapes
:
£ђЛ
Read_70/DisableCopyOnReadDisableCopyOnRead6read_70_disablecopyonread_adam_v_food_embedding_kernel"/device:CPU:0*
_output_shapes
 Ї
Read_70/ReadVariableOpReadVariableOp6read_70_disablecopyonread_adam_v_food_embedding_kernel^Read_70/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
£ђ*
dtype0r
Identity_140IdentityRead_70/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
£ђi
Identity_141IdentityIdentity_140:output:0"/device:CPU:0*
T0* 
_output_shapes
:
£ђЙ
Read_71/DisableCopyOnReadDisableCopyOnRead4read_71_disablecopyonread_adam_m_food_embedding_bias"/device:CPU:0*
_output_shapes
 ≥
Read_71/ReadVariableOpReadVariableOp4read_71_disablecopyonread_adam_m_food_embedding_bias^Read_71/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:ђ*
dtype0m
Identity_142IdentityRead_71/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:ђd
Identity_143IdentityIdentity_142:output:0"/device:CPU:0*
T0*
_output_shapes	
:ђЙ
Read_72/DisableCopyOnReadDisableCopyOnRead4read_72_disablecopyonread_adam_v_food_embedding_bias"/device:CPU:0*
_output_shapes
 ≥
Read_72/ReadVariableOpReadVariableOp4read_72_disablecopyonread_adam_v_food_embedding_bias^Read_72/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:ђ*
dtype0m
Identity_144IdentityRead_72/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:ђd
Identity_145IdentityIdentity_144:output:0"/device:CPU:0*
T0*
_output_shapes	
:ђО
Read_73/DisableCopyOnReadDisableCopyOnRead9read_73_disablecopyonread_adam_m_context_embedding_kernel"/device:CPU:0*
_output_shapes
 Љ
Read_73/ReadVariableOpReadVariableOp9read_73_disablecopyonread_adam_m_context_embedding_kernel^Read_73/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	Ш*
dtype0q
Identity_146IdentityRead_73/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	Шh
Identity_147IdentityIdentity_146:output:0"/device:CPU:0*
T0*
_output_shapes
:	ШО
Read_74/DisableCopyOnReadDisableCopyOnRead9read_74_disablecopyonread_adam_v_context_embedding_kernel"/device:CPU:0*
_output_shapes
 Љ
Read_74/ReadVariableOpReadVariableOp9read_74_disablecopyonread_adam_v_context_embedding_kernel^Read_74/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	Ш*
dtype0q
Identity_148IdentityRead_74/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	Шh
Identity_149IdentityIdentity_148:output:0"/device:CPU:0*
T0*
_output_shapes
:	ШМ
Read_75/DisableCopyOnReadDisableCopyOnRead7read_75_disablecopyonread_adam_m_context_embedding_bias"/device:CPU:0*
_output_shapes
 µ
Read_75/ReadVariableOpReadVariableOp7read_75_disablecopyonread_adam_m_context_embedding_bias^Read_75/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_150IdentityRead_75/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_151IdentityIdentity_150:output:0"/device:CPU:0*
T0*
_output_shapes
:М
Read_76/DisableCopyOnReadDisableCopyOnRead7read_76_disablecopyonread_adam_v_context_embedding_bias"/device:CPU:0*
_output_shapes
 µ
Read_76/ReadVariableOpReadVariableOp7read_76_disablecopyonread_adam_v_context_embedding_bias^Read_76/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_152IdentityRead_76/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_153IdentityIdentity_152:output:0"/device:CPU:0*
T0*
_output_shapes
:Т
Read_77/DisableCopyOnReadDisableCopyOnRead=read_77_disablecopyonread_adam_m_batch_normalization_12_gamma"/device:CPU:0*
_output_shapes
 Љ
Read_77/ReadVariableOpReadVariableOp=read_77_disablecopyonread_adam_m_batch_normalization_12_gamma^Read_77/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:й*
dtype0m
Identity_154IdentityRead_77/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:йd
Identity_155IdentityIdentity_154:output:0"/device:CPU:0*
T0*
_output_shapes	
:йТ
Read_78/DisableCopyOnReadDisableCopyOnRead=read_78_disablecopyonread_adam_v_batch_normalization_12_gamma"/device:CPU:0*
_output_shapes
 Љ
Read_78/ReadVariableOpReadVariableOp=read_78_disablecopyonread_adam_v_batch_normalization_12_gamma^Read_78/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:й*
dtype0m
Identity_156IdentityRead_78/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:йd
Identity_157IdentityIdentity_156:output:0"/device:CPU:0*
T0*
_output_shapes	
:йС
Read_79/DisableCopyOnReadDisableCopyOnRead<read_79_disablecopyonread_adam_m_batch_normalization_12_beta"/device:CPU:0*
_output_shapes
 ї
Read_79/ReadVariableOpReadVariableOp<read_79_disablecopyonread_adam_m_batch_normalization_12_beta^Read_79/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:й*
dtype0m
Identity_158IdentityRead_79/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:йd
Identity_159IdentityIdentity_158:output:0"/device:CPU:0*
T0*
_output_shapes	
:йС
Read_80/DisableCopyOnReadDisableCopyOnRead<read_80_disablecopyonread_adam_v_batch_normalization_12_beta"/device:CPU:0*
_output_shapes
 ї
Read_80/ReadVariableOpReadVariableOp<read_80_disablecopyonread_adam_v_batch_normalization_12_beta^Read_80/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:й*
dtype0m
Identity_160IdentityRead_80/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:йd
Identity_161IdentityIdentity_160:output:0"/device:CPU:0*
T0*
_output_shapes	
:йЗ
Read_81/DisableCopyOnReadDisableCopyOnRead2read_81_disablecopyonread_adam_m_fc_layer_0_kernel"/device:CPU:0*
_output_shapes
 ґ
Read_81/ReadVariableOpReadVariableOp2read_81_disablecopyonread_adam_m_fc_layer_0_kernel^Read_81/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
йА*
dtype0r
Identity_162IdentityRead_81/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
йАi
Identity_163IdentityIdentity_162:output:0"/device:CPU:0*
T0* 
_output_shapes
:
йАЗ
Read_82/DisableCopyOnReadDisableCopyOnRead2read_82_disablecopyonread_adam_v_fc_layer_0_kernel"/device:CPU:0*
_output_shapes
 ґ
Read_82/ReadVariableOpReadVariableOp2read_82_disablecopyonread_adam_v_fc_layer_0_kernel^Read_82/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
йА*
dtype0r
Identity_164IdentityRead_82/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
йАi
Identity_165IdentityIdentity_164:output:0"/device:CPU:0*
T0* 
_output_shapes
:
йАЕ
Read_83/DisableCopyOnReadDisableCopyOnRead0read_83_disablecopyonread_adam_m_fc_layer_0_bias"/device:CPU:0*
_output_shapes
 ѓ
Read_83/ReadVariableOpReadVariableOp0read_83_disablecopyonread_adam_m_fc_layer_0_bias^Read_83/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0m
Identity_166IdentityRead_83/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аd
Identity_167IdentityIdentity_166:output:0"/device:CPU:0*
T0*
_output_shapes	
:АЕ
Read_84/DisableCopyOnReadDisableCopyOnRead0read_84_disablecopyonread_adam_v_fc_layer_0_bias"/device:CPU:0*
_output_shapes
 ѓ
Read_84/ReadVariableOpReadVariableOp0read_84_disablecopyonread_adam_v_fc_layer_0_bias^Read_84/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:А*
dtype0m
Identity_168IdentityRead_84/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Аd
Identity_169IdentityIdentity_168:output:0"/device:CPU:0*
T0*
_output_shapes	
:АЗ
Read_85/DisableCopyOnReadDisableCopyOnRead2read_85_disablecopyonread_adam_m_fc_layer_1_kernel"/device:CPU:0*
_output_shapes
 µ
Read_85/ReadVariableOpReadVariableOp2read_85_disablecopyonread_adam_m_fc_layer_1_kernel^Read_85/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	А@*
dtype0q
Identity_170IdentityRead_85/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	А@h
Identity_171IdentityIdentity_170:output:0"/device:CPU:0*
T0*
_output_shapes
:	А@З
Read_86/DisableCopyOnReadDisableCopyOnRead2read_86_disablecopyonread_adam_v_fc_layer_1_kernel"/device:CPU:0*
_output_shapes
 µ
Read_86/ReadVariableOpReadVariableOp2read_86_disablecopyonread_adam_v_fc_layer_1_kernel^Read_86/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	А@*
dtype0q
Identity_172IdentityRead_86/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	А@h
Identity_173IdentityIdentity_172:output:0"/device:CPU:0*
T0*
_output_shapes
:	А@Е
Read_87/DisableCopyOnReadDisableCopyOnRead0read_87_disablecopyonread_adam_m_fc_layer_1_bias"/device:CPU:0*
_output_shapes
 Ѓ
Read_87/ReadVariableOpReadVariableOp0read_87_disablecopyonread_adam_m_fc_layer_1_bias^Read_87/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_174IdentityRead_87/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_175IdentityIdentity_174:output:0"/device:CPU:0*
T0*
_output_shapes
:@Е
Read_88/DisableCopyOnReadDisableCopyOnRead0read_88_disablecopyonread_adam_v_fc_layer_1_bias"/device:CPU:0*
_output_shapes
 Ѓ
Read_88/ReadVariableOpReadVariableOp0read_88_disablecopyonread_adam_v_fc_layer_1_bias^Read_88/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_176IdentityRead_88/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_177IdentityIdentity_176:output:0"/device:CPU:0*
T0*
_output_shapes
:@З
Read_89/DisableCopyOnReadDisableCopyOnRead2read_89_disablecopyonread_adam_m_fc_layer_2_kernel"/device:CPU:0*
_output_shapes
 і
Read_89/ReadVariableOpReadVariableOp2read_89_disablecopyonread_adam_m_fc_layer_2_kernel^Read_89/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@ *
dtype0p
Identity_178IdentityRead_89/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@ g
Identity_179IdentityIdentity_178:output:0"/device:CPU:0*
T0*
_output_shapes

:@ З
Read_90/DisableCopyOnReadDisableCopyOnRead2read_90_disablecopyonread_adam_v_fc_layer_2_kernel"/device:CPU:0*
_output_shapes
 і
Read_90/ReadVariableOpReadVariableOp2read_90_disablecopyonread_adam_v_fc_layer_2_kernel^Read_90/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@ *
dtype0p
Identity_180IdentityRead_90/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@ g
Identity_181IdentityIdentity_180:output:0"/device:CPU:0*
T0*
_output_shapes

:@ Е
Read_91/DisableCopyOnReadDisableCopyOnRead0read_91_disablecopyonread_adam_m_fc_layer_2_bias"/device:CPU:0*
_output_shapes
 Ѓ
Read_91/ReadVariableOpReadVariableOp0read_91_disablecopyonread_adam_m_fc_layer_2_bias^Read_91/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0l
Identity_182IdentityRead_91/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_183IdentityIdentity_182:output:0"/device:CPU:0*
T0*
_output_shapes
: Е
Read_92/DisableCopyOnReadDisableCopyOnRead0read_92_disablecopyonread_adam_v_fc_layer_2_bias"/device:CPU:0*
_output_shapes
 Ѓ
Read_92/ReadVariableOpReadVariableOp0read_92_disablecopyonread_adam_v_fc_layer_2_bias^Read_92/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0l
Identity_184IdentityRead_92/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: c
Identity_185IdentityIdentity_184:output:0"/device:CPU:0*
T0*
_output_shapes
: Е
Read_93/DisableCopyOnReadDisableCopyOnRead0read_93_disablecopyonread_adam_m_output_0_kernel"/device:CPU:0*
_output_shapes
 ≤
Read_93/ReadVariableOpReadVariableOp0read_93_disablecopyonread_adam_m_output_0_kernel^Read_93/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0p
Identity_186IdentityRead_93/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: g
Identity_187IdentityIdentity_186:output:0"/device:CPU:0*
T0*
_output_shapes

: Е
Read_94/DisableCopyOnReadDisableCopyOnRead0read_94_disablecopyonread_adam_v_output_0_kernel"/device:CPU:0*
_output_shapes
 ≤
Read_94/ReadVariableOpReadVariableOp0read_94_disablecopyonread_adam_v_output_0_kernel^Read_94/DisableCopyOnRead"/device:CPU:0*
_output_shapes

: *
dtype0p
Identity_188IdentityRead_94/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

: g
Identity_189IdentityIdentity_188:output:0"/device:CPU:0*
T0*
_output_shapes

: Г
Read_95/DisableCopyOnReadDisableCopyOnRead.read_95_disablecopyonread_adam_m_output_0_bias"/device:CPU:0*
_output_shapes
 ђ
Read_95/ReadVariableOpReadVariableOp.read_95_disablecopyonread_adam_m_output_0_bias^Read_95/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_190IdentityRead_95/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_191IdentityIdentity_190:output:0"/device:CPU:0*
T0*
_output_shapes
:Г
Read_96/DisableCopyOnReadDisableCopyOnRead.read_96_disablecopyonread_adam_v_output_0_bias"/device:CPU:0*
_output_shapes
 ђ
Read_96/ReadVariableOpReadVariableOp.read_96_disablecopyonread_adam_v_output_0_bias^Read_96/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_192IdentityRead_96/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_193IdentityIdentity_192:output:0"/device:CPU:0*
T0*
_output_shapes
:v
Read_97/DisableCopyOnReadDisableCopyOnRead!read_97_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 Ы
Read_97/ReadVariableOpReadVariableOp!read_97_disablecopyonread_total_1^Read_97/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_194IdentityRead_97/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_195IdentityIdentity_194:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_98/DisableCopyOnReadDisableCopyOnRead!read_98_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 Ы
Read_98/ReadVariableOpReadVariableOp!read_98_disablecopyonread_count_1^Read_98/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_196IdentityRead_98/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_197IdentityIdentity_196:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_99/DisableCopyOnReadDisableCopyOnReadread_99_disablecopyonread_total"/device:CPU:0*
_output_shapes
 Щ
Read_99/ReadVariableOpReadVariableOpread_99_disablecopyonread_total^Read_99/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_198IdentityRead_99/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_199IdentityIdentity_198:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_100/DisableCopyOnReadDisableCopyOnRead read_100_disablecopyonread_count"/device:CPU:0*
_output_shapes
 Ь
Read_100/ReadVariableOpReadVariableOp read_100_disablecopyonread_count^Read_100/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i
Identity_200IdentityRead_100/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_201IdentityIdentity_200:output:0"/device:CPU:0*
T0*
_output_shapes
: Б
Read_101/DisableCopyOnReadDisableCopyOnRead+read_101_disablecopyonread_true_positives_1"/device:CPU:0*
_output_shapes
 Ђ
Read_101/ReadVariableOpReadVariableOp+read_101_disablecopyonread_true_positives_1^Read_101/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_202IdentityRead_101/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_203IdentityIdentity_202:output:0"/device:CPU:0*
T0*
_output_shapes
:А
Read_102/DisableCopyOnReadDisableCopyOnRead*read_102_disablecopyonread_false_positives"/device:CPU:0*
_output_shapes
 ™
Read_102/ReadVariableOpReadVariableOp*read_102_disablecopyonread_false_positives^Read_102/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_204IdentityRead_102/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_205IdentityIdentity_204:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_103/DisableCopyOnReadDisableCopyOnRead)read_103_disablecopyonread_true_positives"/device:CPU:0*
_output_shapes
 ©
Read_103/ReadVariableOpReadVariableOp)read_103_disablecopyonread_true_positives^Read_103/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_206IdentityRead_103/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_207IdentityIdentity_206:output:0"/device:CPU:0*
T0*
_output_shapes
:А
Read_104/DisableCopyOnReadDisableCopyOnRead*read_104_disablecopyonread_false_negatives"/device:CPU:0*
_output_shapes
 ™
Read_104/ReadVariableOpReadVariableOp*read_104_disablecopyonread_false_negatives^Read_104/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0m
Identity_208IdentityRead_104/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_209IdentityIdentity_208:output:0"/device:CPU:0*
T0*
_output_shapes
:ё-
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:j*
dtype0*З-
valueэ,Bъ,jB:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-2/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-3/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-4/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-5/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-6/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-7/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-8/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-9/embeddings/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-10/embeddings/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-11/embeddings/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-12/embeddings/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-13/embeddings/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-14/embeddings/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-18/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-18/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-18/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-22/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/49/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/50/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/51/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/52/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/53/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/54/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/55/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/56/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/57/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/58/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/59/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/60/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/61/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/62/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHƒ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:j*
dtype0*й
valueяB№jB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ь
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0Identity_85:output:0Identity_87:output:0Identity_89:output:0Identity_91:output:0Identity_93:output:0Identity_95:output:0Identity_97:output:0Identity_99:output:0Identity_101:output:0Identity_103:output:0Identity_105:output:0Identity_107:output:0Identity_109:output:0Identity_111:output:0Identity_113:output:0Identity_115:output:0Identity_117:output:0Identity_119:output:0Identity_121:output:0Identity_123:output:0Identity_125:output:0Identity_127:output:0Identity_129:output:0Identity_131:output:0Identity_133:output:0Identity_135:output:0Identity_137:output:0Identity_139:output:0Identity_141:output:0Identity_143:output:0Identity_145:output:0Identity_147:output:0Identity_149:output:0Identity_151:output:0Identity_153:output:0Identity_155:output:0Identity_157:output:0Identity_159:output:0Identity_161:output:0Identity_163:output:0Identity_165:output:0Identity_167:output:0Identity_169:output:0Identity_171:output:0Identity_173:output:0Identity_175:output:0Identity_177:output:0Identity_179:output:0Identity_181:output:0Identity_183:output:0Identity_185:output:0Identity_187:output:0Identity_189:output:0Identity_191:output:0Identity_193:output:0Identity_195:output:0Identity_197:output:0Identity_199:output:0Identity_201:output:0Identity_203:output:0Identity_205:output:0Identity_207:output:0Identity_209:output:0savev2_const_51"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *x
dtypesn
l2j	Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:≥
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 j
Identity_210Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: W
Identity_211IdentityIdentity_210:output:0^NoOp*
T0*
_output_shapes
: ж+
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_100/DisableCopyOnRead^Read_100/ReadVariableOp^Read_101/DisableCopyOnRead^Read_101/ReadVariableOp^Read_102/DisableCopyOnRead^Read_102/ReadVariableOp^Read_103/DisableCopyOnRead^Read_103/ReadVariableOp^Read_104/DisableCopyOnRead^Read_104/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_42/DisableCopyOnRead^Read_42/ReadVariableOp^Read_43/DisableCopyOnRead^Read_43/ReadVariableOp^Read_44/DisableCopyOnRead^Read_44/ReadVariableOp^Read_45/DisableCopyOnRead^Read_45/ReadVariableOp^Read_46/DisableCopyOnRead^Read_46/ReadVariableOp^Read_47/DisableCopyOnRead^Read_47/ReadVariableOp^Read_48/DisableCopyOnRead^Read_48/ReadVariableOp^Read_49/DisableCopyOnRead^Read_49/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_50/DisableCopyOnRead^Read_50/ReadVariableOp^Read_51/DisableCopyOnRead^Read_51/ReadVariableOp^Read_52/DisableCopyOnRead^Read_52/ReadVariableOp^Read_53/DisableCopyOnRead^Read_53/ReadVariableOp^Read_54/DisableCopyOnRead^Read_54/ReadVariableOp^Read_55/DisableCopyOnRead^Read_55/ReadVariableOp^Read_56/DisableCopyOnRead^Read_56/ReadVariableOp^Read_57/DisableCopyOnRead^Read_57/ReadVariableOp^Read_58/DisableCopyOnRead^Read_58/ReadVariableOp^Read_59/DisableCopyOnRead^Read_59/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_60/DisableCopyOnRead^Read_60/ReadVariableOp^Read_61/DisableCopyOnRead^Read_61/ReadVariableOp^Read_62/DisableCopyOnRead^Read_62/ReadVariableOp^Read_63/DisableCopyOnRead^Read_63/ReadVariableOp^Read_64/DisableCopyOnRead^Read_64/ReadVariableOp^Read_65/DisableCopyOnRead^Read_65/ReadVariableOp^Read_66/DisableCopyOnRead^Read_66/ReadVariableOp^Read_67/DisableCopyOnRead^Read_67/ReadVariableOp^Read_68/DisableCopyOnRead^Read_68/ReadVariableOp^Read_69/DisableCopyOnRead^Read_69/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_70/DisableCopyOnRead^Read_70/ReadVariableOp^Read_71/DisableCopyOnRead^Read_71/ReadVariableOp^Read_72/DisableCopyOnRead^Read_72/ReadVariableOp^Read_73/DisableCopyOnRead^Read_73/ReadVariableOp^Read_74/DisableCopyOnRead^Read_74/ReadVariableOp^Read_75/DisableCopyOnRead^Read_75/ReadVariableOp^Read_76/DisableCopyOnRead^Read_76/ReadVariableOp^Read_77/DisableCopyOnRead^Read_77/ReadVariableOp^Read_78/DisableCopyOnRead^Read_78/ReadVariableOp^Read_79/DisableCopyOnRead^Read_79/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_80/DisableCopyOnRead^Read_80/ReadVariableOp^Read_81/DisableCopyOnRead^Read_81/ReadVariableOp^Read_82/DisableCopyOnRead^Read_82/ReadVariableOp^Read_83/DisableCopyOnRead^Read_83/ReadVariableOp^Read_84/DisableCopyOnRead^Read_84/ReadVariableOp^Read_85/DisableCopyOnRead^Read_85/ReadVariableOp^Read_86/DisableCopyOnRead^Read_86/ReadVariableOp^Read_87/DisableCopyOnRead^Read_87/ReadVariableOp^Read_88/DisableCopyOnRead^Read_88/ReadVariableOp^Read_89/DisableCopyOnRead^Read_89/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp^Read_90/DisableCopyOnRead^Read_90/ReadVariableOp^Read_91/DisableCopyOnRead^Read_91/ReadVariableOp^Read_92/DisableCopyOnRead^Read_92/ReadVariableOp^Read_93/DisableCopyOnRead^Read_93/ReadVariableOp^Read_94/DisableCopyOnRead^Read_94/ReadVariableOp^Read_95/DisableCopyOnRead^Read_95/ReadVariableOp^Read_96/DisableCopyOnRead^Read_96/ReadVariableOp^Read_97/DisableCopyOnRead^Read_97/ReadVariableOp^Read_98/DisableCopyOnRead^Read_98/ReadVariableOp^Read_99/DisableCopyOnRead^Read_99/ReadVariableOp*
_output_shapes
 "%
identity_211Identity_211:output:0*(
_construction_contextkEagerRuntime*л
_input_shapesў
÷: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp28
Read_100/DisableCopyOnReadRead_100/DisableCopyOnRead22
Read_100/ReadVariableOpRead_100/ReadVariableOp28
Read_101/DisableCopyOnReadRead_101/DisableCopyOnRead22
Read_101/ReadVariableOpRead_101/ReadVariableOp28
Read_102/DisableCopyOnReadRead_102/DisableCopyOnRead22
Read_102/ReadVariableOpRead_102/ReadVariableOp28
Read_103/DisableCopyOnReadRead_103/DisableCopyOnRead22
Read_103/ReadVariableOpRead_103/ReadVariableOp28
Read_104/DisableCopyOnReadRead_104/DisableCopyOnRead22
Read_104/ReadVariableOpRead_104/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp26
Read_36/DisableCopyOnReadRead_36/DisableCopyOnRead20
Read_36/ReadVariableOpRead_36/ReadVariableOp26
Read_37/DisableCopyOnReadRead_37/DisableCopyOnRead20
Read_37/ReadVariableOpRead_37/ReadVariableOp26
Read_38/DisableCopyOnReadRead_38/DisableCopyOnRead20
Read_38/ReadVariableOpRead_38/ReadVariableOp26
Read_39/DisableCopyOnReadRead_39/DisableCopyOnRead20
Read_39/ReadVariableOpRead_39/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp26
Read_40/DisableCopyOnReadRead_40/DisableCopyOnRead20
Read_40/ReadVariableOpRead_40/ReadVariableOp26
Read_41/DisableCopyOnReadRead_41/DisableCopyOnRead20
Read_41/ReadVariableOpRead_41/ReadVariableOp26
Read_42/DisableCopyOnReadRead_42/DisableCopyOnRead20
Read_42/ReadVariableOpRead_42/ReadVariableOp26
Read_43/DisableCopyOnReadRead_43/DisableCopyOnRead20
Read_43/ReadVariableOpRead_43/ReadVariableOp26
Read_44/DisableCopyOnReadRead_44/DisableCopyOnRead20
Read_44/ReadVariableOpRead_44/ReadVariableOp26
Read_45/DisableCopyOnReadRead_45/DisableCopyOnRead20
Read_45/ReadVariableOpRead_45/ReadVariableOp26
Read_46/DisableCopyOnReadRead_46/DisableCopyOnRead20
Read_46/ReadVariableOpRead_46/ReadVariableOp26
Read_47/DisableCopyOnReadRead_47/DisableCopyOnRead20
Read_47/ReadVariableOpRead_47/ReadVariableOp26
Read_48/DisableCopyOnReadRead_48/DisableCopyOnRead20
Read_48/ReadVariableOpRead_48/ReadVariableOp26
Read_49/DisableCopyOnReadRead_49/DisableCopyOnRead20
Read_49/ReadVariableOpRead_49/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp26
Read_50/DisableCopyOnReadRead_50/DisableCopyOnRead20
Read_50/ReadVariableOpRead_50/ReadVariableOp26
Read_51/DisableCopyOnReadRead_51/DisableCopyOnRead20
Read_51/ReadVariableOpRead_51/ReadVariableOp26
Read_52/DisableCopyOnReadRead_52/DisableCopyOnRead20
Read_52/ReadVariableOpRead_52/ReadVariableOp26
Read_53/DisableCopyOnReadRead_53/DisableCopyOnRead20
Read_53/ReadVariableOpRead_53/ReadVariableOp26
Read_54/DisableCopyOnReadRead_54/DisableCopyOnRead20
Read_54/ReadVariableOpRead_54/ReadVariableOp26
Read_55/DisableCopyOnReadRead_55/DisableCopyOnRead20
Read_55/ReadVariableOpRead_55/ReadVariableOp26
Read_56/DisableCopyOnReadRead_56/DisableCopyOnRead20
Read_56/ReadVariableOpRead_56/ReadVariableOp26
Read_57/DisableCopyOnReadRead_57/DisableCopyOnRead20
Read_57/ReadVariableOpRead_57/ReadVariableOp26
Read_58/DisableCopyOnReadRead_58/DisableCopyOnRead20
Read_58/ReadVariableOpRead_58/ReadVariableOp26
Read_59/DisableCopyOnReadRead_59/DisableCopyOnRead20
Read_59/ReadVariableOpRead_59/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp26
Read_60/DisableCopyOnReadRead_60/DisableCopyOnRead20
Read_60/ReadVariableOpRead_60/ReadVariableOp26
Read_61/DisableCopyOnReadRead_61/DisableCopyOnRead20
Read_61/ReadVariableOpRead_61/ReadVariableOp26
Read_62/DisableCopyOnReadRead_62/DisableCopyOnRead20
Read_62/ReadVariableOpRead_62/ReadVariableOp26
Read_63/DisableCopyOnReadRead_63/DisableCopyOnRead20
Read_63/ReadVariableOpRead_63/ReadVariableOp26
Read_64/DisableCopyOnReadRead_64/DisableCopyOnRead20
Read_64/ReadVariableOpRead_64/ReadVariableOp26
Read_65/DisableCopyOnReadRead_65/DisableCopyOnRead20
Read_65/ReadVariableOpRead_65/ReadVariableOp26
Read_66/DisableCopyOnReadRead_66/DisableCopyOnRead20
Read_66/ReadVariableOpRead_66/ReadVariableOp26
Read_67/DisableCopyOnReadRead_67/DisableCopyOnRead20
Read_67/ReadVariableOpRead_67/ReadVariableOp26
Read_68/DisableCopyOnReadRead_68/DisableCopyOnRead20
Read_68/ReadVariableOpRead_68/ReadVariableOp26
Read_69/DisableCopyOnReadRead_69/DisableCopyOnRead20
Read_69/ReadVariableOpRead_69/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp26
Read_70/DisableCopyOnReadRead_70/DisableCopyOnRead20
Read_70/ReadVariableOpRead_70/ReadVariableOp26
Read_71/DisableCopyOnReadRead_71/DisableCopyOnRead20
Read_71/ReadVariableOpRead_71/ReadVariableOp26
Read_72/DisableCopyOnReadRead_72/DisableCopyOnRead20
Read_72/ReadVariableOpRead_72/ReadVariableOp26
Read_73/DisableCopyOnReadRead_73/DisableCopyOnRead20
Read_73/ReadVariableOpRead_73/ReadVariableOp26
Read_74/DisableCopyOnReadRead_74/DisableCopyOnRead20
Read_74/ReadVariableOpRead_74/ReadVariableOp26
Read_75/DisableCopyOnReadRead_75/DisableCopyOnRead20
Read_75/ReadVariableOpRead_75/ReadVariableOp26
Read_76/DisableCopyOnReadRead_76/DisableCopyOnRead20
Read_76/ReadVariableOpRead_76/ReadVariableOp26
Read_77/DisableCopyOnReadRead_77/DisableCopyOnRead20
Read_77/ReadVariableOpRead_77/ReadVariableOp26
Read_78/DisableCopyOnReadRead_78/DisableCopyOnRead20
Read_78/ReadVariableOpRead_78/ReadVariableOp26
Read_79/DisableCopyOnReadRead_79/DisableCopyOnRead20
Read_79/ReadVariableOpRead_79/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp26
Read_80/DisableCopyOnReadRead_80/DisableCopyOnRead20
Read_80/ReadVariableOpRead_80/ReadVariableOp26
Read_81/DisableCopyOnReadRead_81/DisableCopyOnRead20
Read_81/ReadVariableOpRead_81/ReadVariableOp26
Read_82/DisableCopyOnReadRead_82/DisableCopyOnRead20
Read_82/ReadVariableOpRead_82/ReadVariableOp26
Read_83/DisableCopyOnReadRead_83/DisableCopyOnRead20
Read_83/ReadVariableOpRead_83/ReadVariableOp26
Read_84/DisableCopyOnReadRead_84/DisableCopyOnRead20
Read_84/ReadVariableOpRead_84/ReadVariableOp26
Read_85/DisableCopyOnReadRead_85/DisableCopyOnRead20
Read_85/ReadVariableOpRead_85/ReadVariableOp26
Read_86/DisableCopyOnReadRead_86/DisableCopyOnRead20
Read_86/ReadVariableOpRead_86/ReadVariableOp26
Read_87/DisableCopyOnReadRead_87/DisableCopyOnRead20
Read_87/ReadVariableOpRead_87/ReadVariableOp26
Read_88/DisableCopyOnReadRead_88/DisableCopyOnRead20
Read_88/ReadVariableOpRead_88/ReadVariableOp26
Read_89/DisableCopyOnReadRead_89/DisableCopyOnRead20
Read_89/ReadVariableOpRead_89/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp26
Read_90/DisableCopyOnReadRead_90/DisableCopyOnRead20
Read_90/ReadVariableOpRead_90/ReadVariableOp26
Read_91/DisableCopyOnReadRead_91/DisableCopyOnRead20
Read_91/ReadVariableOpRead_91/ReadVariableOp26
Read_92/DisableCopyOnReadRead_92/DisableCopyOnRead20
Read_92/ReadVariableOpRead_92/ReadVariableOp26
Read_93/DisableCopyOnReadRead_93/DisableCopyOnRead20
Read_93/ReadVariableOpRead_93/ReadVariableOp26
Read_94/DisableCopyOnReadRead_94/DisableCopyOnRead20
Read_94/ReadVariableOpRead_94/ReadVariableOp26
Read_95/DisableCopyOnReadRead_95/DisableCopyOnRead20
Read_95/ReadVariableOpRead_95/ReadVariableOp26
Read_96/DisableCopyOnReadRead_96/DisableCopyOnRead20
Read_96/ReadVariableOpRead_96/ReadVariableOp26
Read_97/DisableCopyOnReadRead_97/DisableCopyOnRead20
Read_97/ReadVariableOpRead_97/ReadVariableOp26
Read_98/DisableCopyOnReadRead_98/DisableCopyOnRead20
Read_98/ReadVariableOpRead_98/ReadVariableOp26
Read_99/DisableCopyOnReadRead_99/DisableCopyOnRead20
Read_99/ReadVariableOpRead_99/ReadVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:73
1
_user_specified_nameembedding_76/embeddings:73
1
_user_specified_nameembedding_77/embeddings:73
1
_user_specified_nameembedding_78/embeddings:73
1
_user_specified_nameembedding_79/embeddings:73
1
_user_specified_nameembedding_80/embeddings:73
1
_user_specified_nameembedding_81/embeddings:73
1
_user_specified_nameembedding_82/embeddings:73
1
_user_specified_nameembedding_83/embeddings:7	3
1
_user_specified_nameembedding_84/embeddings:7
3
1
_user_specified_nameembedding_85/embeddings:73
1
_user_specified_nameembedding_86/embeddings:73
1
_user_specified_nameembedding_89/embeddings:73
1
_user_specified_nameembedding_90/embeddings:73
1
_user_specified_nameembedding_87/embeddings:73
1
_user_specified_nameembedding_88/embeddings:51
/
_user_specified_nameuser_embedding/kernel:3/
-
_user_specified_nameuser_embedding/bias:51
/
_user_specified_namefood_embedding/kernel:3/
-
_user_specified_namefood_embedding/bias:84
2
_user_specified_namecontext_embedding/kernel:62
0
_user_specified_namecontext_embedding/bias:<8
6
_user_specified_namebatch_normalization_12/gamma:;7
5
_user_specified_namebatch_normalization_12/beta:B>
<
_user_specified_name$"batch_normalization_12/moving_mean:FB
@
_user_specified_name(&batch_normalization_12/moving_variance:1-
+
_user_specified_namefc_layer_0/kernel:/+
)
_user_specified_namefc_layer_0/bias:1-
+
_user_specified_namefc_layer_1/kernel:/+
)
_user_specified_namefc_layer_1/bias:1-
+
_user_specified_namefc_layer_2/kernel:/+
)
_user_specified_namefc_layer_2/bias:/ +
)
_user_specified_nameoutput_0/kernel:-!)
'
_user_specified_nameoutput_0/bias:)"%
#
_user_specified_name	iteration:-#)
'
_user_specified_namelearning_rate:>$:
8
_user_specified_name Adam/m/embedding_76/embeddings:>%:
8
_user_specified_name Adam/v/embedding_76/embeddings:>&:
8
_user_specified_name Adam/m/embedding_77/embeddings:>':
8
_user_specified_name Adam/v/embedding_77/embeddings:>(:
8
_user_specified_name Adam/m/embedding_78/embeddings:>):
8
_user_specified_name Adam/v/embedding_78/embeddings:>*:
8
_user_specified_name Adam/m/embedding_79/embeddings:>+:
8
_user_specified_name Adam/v/embedding_79/embeddings:>,:
8
_user_specified_name Adam/m/embedding_80/embeddings:>-:
8
_user_specified_name Adam/v/embedding_80/embeddings:>.:
8
_user_specified_name Adam/m/embedding_81/embeddings:>/:
8
_user_specified_name Adam/v/embedding_81/embeddings:>0:
8
_user_specified_name Adam/m/embedding_82/embeddings:>1:
8
_user_specified_name Adam/v/embedding_82/embeddings:>2:
8
_user_specified_name Adam/m/embedding_83/embeddings:>3:
8
_user_specified_name Adam/v/embedding_83/embeddings:>4:
8
_user_specified_name Adam/m/embedding_84/embeddings:>5:
8
_user_specified_name Adam/v/embedding_84/embeddings:>6:
8
_user_specified_name Adam/m/embedding_85/embeddings:>7:
8
_user_specified_name Adam/v/embedding_85/embeddings:>8:
8
_user_specified_name Adam/m/embedding_86/embeddings:>9:
8
_user_specified_name Adam/v/embedding_86/embeddings:>::
8
_user_specified_name Adam/m/embedding_89/embeddings:>;:
8
_user_specified_name Adam/v/embedding_89/embeddings:><:
8
_user_specified_name Adam/m/embedding_90/embeddings:>=:
8
_user_specified_name Adam/v/embedding_90/embeddings:>>:
8
_user_specified_name Adam/m/embedding_87/embeddings:>?:
8
_user_specified_name Adam/v/embedding_87/embeddings:>@:
8
_user_specified_name Adam/m/embedding_88/embeddings:>A:
8
_user_specified_name Adam/v/embedding_88/embeddings:<B8
6
_user_specified_nameAdam/m/user_embedding/kernel:<C8
6
_user_specified_nameAdam/v/user_embedding/kernel::D6
4
_user_specified_nameAdam/m/user_embedding/bias::E6
4
_user_specified_nameAdam/v/user_embedding/bias:<F8
6
_user_specified_nameAdam/m/food_embedding/kernel:<G8
6
_user_specified_nameAdam/v/food_embedding/kernel::H6
4
_user_specified_nameAdam/m/food_embedding/bias::I6
4
_user_specified_nameAdam/v/food_embedding/bias:?J;
9
_user_specified_name!Adam/m/context_embedding/kernel:?K;
9
_user_specified_name!Adam/v/context_embedding/kernel:=L9
7
_user_specified_nameAdam/m/context_embedding/bias:=M9
7
_user_specified_nameAdam/v/context_embedding/bias:CN?
=
_user_specified_name%#Adam/m/batch_normalization_12/gamma:CO?
=
_user_specified_name%#Adam/v/batch_normalization_12/gamma:BP>
<
_user_specified_name$"Adam/m/batch_normalization_12/beta:BQ>
<
_user_specified_name$"Adam/v/batch_normalization_12/beta:8R4
2
_user_specified_nameAdam/m/fc_layer_0/kernel:8S4
2
_user_specified_nameAdam/v/fc_layer_0/kernel:6T2
0
_user_specified_nameAdam/m/fc_layer_0/bias:6U2
0
_user_specified_nameAdam/v/fc_layer_0/bias:8V4
2
_user_specified_nameAdam/m/fc_layer_1/kernel:8W4
2
_user_specified_nameAdam/v/fc_layer_1/kernel:6X2
0
_user_specified_nameAdam/m/fc_layer_1/bias:6Y2
0
_user_specified_nameAdam/v/fc_layer_1/bias:8Z4
2
_user_specified_nameAdam/m/fc_layer_2/kernel:8[4
2
_user_specified_nameAdam/v/fc_layer_2/kernel:6\2
0
_user_specified_nameAdam/m/fc_layer_2/bias:6]2
0
_user_specified_nameAdam/v/fc_layer_2/bias:6^2
0
_user_specified_nameAdam/m/output_0/kernel:6_2
0
_user_specified_nameAdam/v/output_0/kernel:4`0
.
_user_specified_nameAdam/m/output_0/bias:4a0
.
_user_specified_nameAdam/v/output_0/bias:'b#
!
_user_specified_name	total_1:'c#
!
_user_specified_name	count_1:%d!

_user_specified_nametotal:%e!

_user_specified_namecount:0f,
*
_user_specified_nametrue_positives_1:/g+
)
_user_specified_namefalse_positives:.h*
(
_user_specified_nametrue_positives:/i+
)
_user_specified_namefalse_negatives:@j<

_output_shapes
: 
"
_user_specified_name
Const_51
ѓ
Е
 __inference__initializer_7098181:
6key_value_init6435053_lookuptableimportv2_table_handle2
.key_value_init6435053_lookuptableimportv2_keys4
0key_value_init6435053_lookuptableimportv2_values	
identityИҐ)key_value_init6435053/LookupTableImportV2З
)key_value_init6435053/LookupTableImportV2LookupTableImportV26key_value_init6435053_lookuptableimportv2_table_handle.key_value_init6435053_lookuptableimportv2_keys0key_value_init6435053_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: N
NoOpNoOp*^key_value_init6435053/LookupTableImportV2*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2V
)key_value_init6435053/LookupTableImportV2)key_value_init6435053/LookupTableImportV2:, (
&
_user_specified_nametable_handle: 

_output_shapes
:: 

_output_shapes
:
ВN
–
%__inference_signature_wrapper_7096920
bmi
	age_range
	allergens
allergy
calories
carbohydrates
clinical_gender
cultural_factor
cultural_restriction
current_daily_calories
current_working_status

day_number

embeddings
	ethnicity
fat	
fiber

height

life_style
marital_status
meal_type_y
next_bmi
nutrition_goal
place_of_meal_consumption	
price
projected_daily_calories
protein(
$social_situation_of_meal_consumption	
taste
time_of_meal_consumption

weight
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3
	unknown_4	
	unknown_5
	unknown_6	
	unknown_7
	unknown_8	
	unknown_9

unknown_10	

unknown_11

unknown_12	

unknown_13

unknown_14	

unknown_15

unknown_16	

unknown_17

unknown_18	

unknown_19

unknown_20	

unknown_21

unknown_22	

unknown_23

unknown_24	

unknown_25

unknown_26	

unknown_27

unknown_28	

unknown_29:W	

unknown_30:

unknown_31:

unknown_32:

unknown_33:

unknown_34:

unknown_35:

unknown_36:

unknown_37:

unknown_38:

unknown_39:

unknown_40:

unknown_41:

unknown_42:

unknown_43:

unknown_44

unknown_45	

unknown_46

unknown_47	

unknown_48:	ђ

unknown_49:	ђ

unknown_50:
£ђ

unknown_51:	ђ

unknown_52:	Ш

unknown_53:

unknown_54:	й

unknown_55:	й

unknown_56:	й

unknown_57:	й

unknown_58:
йА

unknown_59:	А

unknown_60:	А@

unknown_61:@

unknown_62:@ 

unknown_63: 

unknown_64: 

unknown_65:
identityИҐStatefulPartitionedCallе
StatefulPartitionedCallStatefulPartitionedCallbmi	age_range	allergensallergycaloriescarbohydratesclinical_gendercultural_factorcultural_restrictioncurrent_daily_caloriescurrent_working_status
day_number
embeddings	ethnicityfatfiberheight
life_stylemarital_statusmeal_type_ynext_bminutrition_goalplace_of_meal_consumptionpriceprojected_daily_caloriesprotein$social_situation_of_meal_consumptiontastetime_of_meal_consumptionweightunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65*l
Tine
c2a																	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*C
_read_only_resource_inputs%
#!<=>?@ABCDEFGHIJOPQRSTUVWXYZ[\]^_`*0
config_proto 

CPU

GPU2*0J 8В *+
f&R$
"__inference__wrapped_model_7094806o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*÷
_input_shapesƒ
Ѕ:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€А:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
'
_output_shapes
:€€€€€€€€€

_user_specified_nameBMI:RN
'
_output_shapes
:€€€€€€€€€
#
_user_specified_name	age_range:RN
'
_output_shapes
:€€€€€€€€€
#
_user_specified_name	allergens:PL
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	allergy:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
calories:VR
'
_output_shapes
:€€€€€€€€€
'
_user_specified_namecarbohydrates:XT
'
_output_shapes
:€€€€€€€€€
)
_user_specified_nameclinical_gender:XT
'
_output_shapes
:€€€€€€€€€
)
_user_specified_namecultural_factor:]Y
'
_output_shapes
:€€€€€€€€€
.
_user_specified_namecultural_restriction:_	[
'
_output_shapes
:€€€€€€€€€
0
_user_specified_namecurrent_daily_calories:_
[
'
_output_shapes
:€€€€€€€€€
0
_user_specified_namecurrent_working_status:SO
'
_output_shapes
:€€€€€€€€€
$
_user_specified_name
day_number:TP
(
_output_shapes
:€€€€€€€€€А
$
_user_specified_name
embeddings:RN
'
_output_shapes
:€€€€€€€€€
#
_user_specified_name	ethnicity:LH
'
_output_shapes
:€€€€€€€€€

_user_specified_namefat:NJ
'
_output_shapes
:€€€€€€€€€

_user_specified_namefiber:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameheight:SO
'
_output_shapes
:€€€€€€€€€
$
_user_specified_name
life_style:WS
'
_output_shapes
:€€€€€€€€€
(
_user_specified_namemarital_status:TP
'
_output_shapes
:€€€€€€€€€
%
_user_specified_namemeal_type_y:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
next_BMI:WS
'
_output_shapes
:€€€€€€€€€
(
_user_specified_namenutrition_goal:b^
'
_output_shapes
:€€€€€€€€€
3
_user_specified_nameplace_of_meal_consumption:NJ
'
_output_shapes
:€€€€€€€€€

_user_specified_nameprice:a]
'
_output_shapes
:€€€€€€€€€
2
_user_specified_nameprojected_daily_calories:PL
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	protein:mi
'
_output_shapes
:€€€€€€€€€
>
_user_specified_name&$social_situation_of_meal_consumption:NJ
'
_output_shapes
:€€€€€€€€€

_user_specified_nametaste:a]
'
_output_shapes
:€€€€€€€€€
2
_user_specified_nametime_of_meal_consumption:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameweight:'#
!
_user_specified_name	7096784:

_output_shapes
: :' #
!
_user_specified_name	7096788:!

_output_shapes
: :'"#
!
_user_specified_name	7096792:#

_output_shapes
: :'$#
!
_user_specified_name	7096796:%

_output_shapes
: :'&#
!
_user_specified_name	7096800:'

_output_shapes
: :'(#
!
_user_specified_name	7096804:)

_output_shapes
: :'*#
!
_user_specified_name	7096808:+

_output_shapes
: :',#
!
_user_specified_name	7096812:-

_output_shapes
: :'.#
!
_user_specified_name	7096816:/

_output_shapes
: :'0#
!
_user_specified_name	7096820:1

_output_shapes
: :'2#
!
_user_specified_name	7096824:3

_output_shapes
: :'4#
!
_user_specified_name	7096828:5

_output_shapes
: :'6#
!
_user_specified_name	7096832:7

_output_shapes
: :'8#
!
_user_specified_name	7096836:9

_output_shapes
: :':#
!
_user_specified_name	7096840:;

_output_shapes
: :'<#
!
_user_specified_name	7096844:'=#
!
_user_specified_name	7096846:'>#
!
_user_specified_name	7096848:'?#
!
_user_specified_name	7096850:'@#
!
_user_specified_name	7096852:'A#
!
_user_specified_name	7096854:'B#
!
_user_specified_name	7096856:'C#
!
_user_specified_name	7096858:'D#
!
_user_specified_name	7096860:'E#
!
_user_specified_name	7096862:'F#
!
_user_specified_name	7096864:'G#
!
_user_specified_name	7096866:'H#
!
_user_specified_name	7096868:'I#
!
_user_specified_name	7096870:'J#
!
_user_specified_name	7096872:'K#
!
_user_specified_name	7096874:L

_output_shapes
: :'M#
!
_user_specified_name	7096878:N

_output_shapes
: :'O#
!
_user_specified_name	7096882:'P#
!
_user_specified_name	7096884:'Q#
!
_user_specified_name	7096886:'R#
!
_user_specified_name	7096888:'S#
!
_user_specified_name	7096890:'T#
!
_user_specified_name	7096892:'U#
!
_user_specified_name	7096894:'V#
!
_user_specified_name	7096896:'W#
!
_user_specified_name	7096898:'X#
!
_user_specified_name	7096900:'Y#
!
_user_specified_name	7096902:'Z#
!
_user_specified_name	7096904:'[#
!
_user_specified_name	7096906:'\#
!
_user_specified_name	7096908:']#
!
_user_specified_name	7096910:'^#
!
_user_specified_name	7096912:'_#
!
_user_specified_name	7096914:'`#
!
_user_specified_name	7096916
ѓ
<
__inference__creator_7098159
identityИҐ
hash_tableo

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	6435003*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: /
NoOpNoOp^hash_table*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
ѓ
Е
 __inference__initializer_7098061:
6key_value_init6434659_lookuptableimportv2_table_handle2
.key_value_init6434659_lookuptableimportv2_keys4
0key_value_init6434659_lookuptableimportv2_values	
identityИҐ)key_value_init6434659/LookupTableImportV2З
)key_value_init6434659/LookupTableImportV2LookupTableImportV26key_value_init6434659_lookuptableimportv2_table_handle.key_value_init6434659_lookuptableimportv2_keys0key_value_init6434659_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: N
NoOpNoOp*^key_value_init6434659/LookupTableImportV2*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2V
)key_value_init6434659/LookupTableImportV2)key_value_init6434659/LookupTableImportV2:, (
&
_user_specified_nametable_handle: 

_output_shapes
:: 

_output_shapes
:
і
В
.__inference_embedding_86_layer_call_fn_7097290

inputs	
unknown:
identityИҐStatefulPartitionedCallЎ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_embedding_86_layer_call_and_return_conditional_losses_7095127s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:€€€€€€€€€: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:'#
!
_user_specified_name	7097286
ѓ
Е
 __inference__initializer_7098001:
6key_value_init6434455_lookuptableimportv2_table_handle2
.key_value_init6434455_lookuptableimportv2_keys4
0key_value_init6434455_lookuptableimportv2_values	
identityИҐ)key_value_init6434455/LookupTableImportV2З
)key_value_init6434455/LookupTableImportV2LookupTableImportV26key_value_init6434455_lookuptableimportv2_table_handle.key_value_init6434455_lookuptableimportv2_keys0key_value_init6434455_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: N
NoOpNoOp*^key_value_init6434455/LookupTableImportV2*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2V
)key_value_init6434455/LookupTableImportV2)key_value_init6434455/LookupTableImportV2:, (
&
_user_specified_nametable_handle: 

_output_shapes
:: 

_output_shapes
:
х
Ч
*__inference_output_0_layer_call_fn_7097889

inputs
unknown: 
	unknown_0:
identityИҐStatefulPartitionedCallЁ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_output_0_layer_call_and_return_conditional_losses_7095647o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs:'#
!
_user_specified_name	7097883:'#
!
_user_specified_name	7097885
њ
R
$__inference__update_step_xla_7097068
gradient
variable:
£ђ*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*!
_input_shapes
:
£ђ: *
	_noinline(:J F
 
_output_shapes
:
£ђ
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
°
І
I__inference_embedding_86_layer_call_and_return_conditional_losses_7095127

inputs	*
embedding_lookup_7095122:
identityИҐembedding_lookupї
embedding_lookupResourceGatherembedding_lookup_7095122inputs*
Tindices0	*+
_class!
loc:@embedding_lookup/7095122*+
_output_shapes
:€€€€€€€€€*
dtype0v
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_output_shapes
:€€€€€€€€€u
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€5
NoOpNoOp^embedding_lookup*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:€€€€€€€€€: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:'#
!
_user_specified_name	7095122
м
±
G__inference_fc_layer_0_layer_call_and_return_conditional_losses_7095587

inputs2
matmul_readvariableop_resource:
йА.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐ3fc_layer_0/kernel/Regularizer/L2Loss/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
йА*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€АФ
3fc_layer_0/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
йА*
dtype0М
$fc_layer_0/kernel/Regularizer/L2LossL2Loss;fc_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#fc_layer_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„£<¶
!fc_layer_0/kernel/Regularizer/mulMul,fc_layer_0/kernel/Regularizer/mul/x:output:0-fc_layer_0/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€АЙ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp4^fc_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€й: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2j
3fc_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp3fc_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€й
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Х
p
$__inference__update_step_xla_7097053
gradient

gradient_1	

gradient_2
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:€€€€€€€€€:€€€€€€€€€:: *
	_noinline(:Q M
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
gradient:MI
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
gradient:D@

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Ѓ
H
,__inference_flatten_76_layer_call_fn_7097333

inputs
identityµ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_flatten_76_layer_call_and_return_conditional_losses_7095309`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ё
Ѓ
G__inference_fc_layer_2_layer_call_and_return_conditional_losses_7095627

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐ3fc_layer_2/kernel/Regularizer/L2Loss/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ Т
3fc_layer_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0М
$fc_layer_2/kernel/Regularizer/L2LossL2Loss;fc_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#fc_layer_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„£<¶
!fc_layer_2/kernel/Regularizer/mulMul,fc_layer_2/kernel/Regularizer/mul/x:output:0-fc_layer_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ Й
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp4^fc_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2j
3fc_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp3fc_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
ѓ
Е
 __inference__initializer_7098046:
6key_value_init6434608_lookuptableimportv2_table_handle2
.key_value_init6434608_lookuptableimportv2_keys4
0key_value_init6434608_lookuptableimportv2_values	
identityИҐ)key_value_init6434608/LookupTableImportV2З
)key_value_init6434608/LookupTableImportV2LookupTableImportV26key_value_init6434608_lookuptableimportv2_table_handle.key_value_init6434608_lookuptableimportv2_keys0key_value_init6434608_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: N
NoOpNoOp*^key_value_init6434608/LookupTableImportV2*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2V
)key_value_init6434608/LookupTableImportV2)key_value_init6434608/LookupTableImportV2:, (
&
_user_specified_nametable_handle: 

_output_shapes
:: 

_output_shapes
:
«
є
K__inference_food_embedding_layer_call_and_return_conditional_losses_7095491

inputs2
matmul_readvariableop_resource:
£ђ.
biasadd_readvariableop_resource:	ђ
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐ7food_embedding/kernel/Regularizer/L2Loss/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
£ђ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђШ
7food_embedding/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
£ђ*
dtype0Ф
(food_embedding/kernel/Regularizer/L2LossL2Loss?food_embedding/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: l
'food_embedding/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„£<≤
%food_embedding/kernel/Regularizer/mulMul0food_embedding/kernel/Regularizer/mul/x:output:01food_embedding/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: `
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€ђН
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp8^food_embedding/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€£: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2r
7food_embedding/kernel/Regularizer/L2Loss/ReadVariableOp7food_embedding/kernel/Regularizer/L2Loss/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€£
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
°
І
I__inference_embedding_77_layer_call_and_return_conditional_losses_7095226

inputs	*
embedding_lookup_7095221:
identityИҐembedding_lookupї
embedding_lookupResourceGatherembedding_lookup_7095221inputs*
Tindices0	*+
_class!
loc:@embedding_lookup/7095221*+
_output_shapes
:€€€€€€€€€*
dtype0v
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_output_shapes
:€€€€€€€€€u
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€5
NoOpNoOp^embedding_lookup*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:€€€€€€€€€: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:'#
!
_user_specified_name	7095221
≠
L
$__inference__update_step_xla_7097083
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
ѓ
<
__inference__creator_7098174
identityИҐ
hash_tableo

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	6435054*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: /
NoOpNoOp^hash_table*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
і
В
.__inference_embedding_77_layer_call_fn_7097155

inputs	
unknown:
identityИҐStatefulPartitionedCallЎ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_embedding_77_layer_call_and_return_conditional_losses_7095226s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:€€€€€€€€€: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:'#
!
_user_specified_name	7097151
Ѓ
H
,__inference_flatten_89_layer_call_fn_7097454

inputs
identityµ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_flatten_89_layer_call_and_return_conditional_losses_7095268`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ѓ
Е
 __inference__initializer_7098151:
6key_value_init6435189_lookuptableimportv2_table_handle2
.key_value_init6435189_lookuptableimportv2_keys4
0key_value_init6435189_lookuptableimportv2_values	
identityИҐ)key_value_init6435189/LookupTableImportV2З
)key_value_init6435189/LookupTableImportV2LookupTableImportV26key_value_init6435189_lookuptableimportv2_table_handle.key_value_init6435189_lookuptableimportv2_keys0key_value_init6435189_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: N
NoOpNoOp*^key_value_init6435189/LookupTableImportV2*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: :W:W2V
)key_value_init6435189/LookupTableImportV2)key_value_init6435189/LookupTableImportV2:, (
&
_user_specified_nametable_handle: 

_output_shapes
:W: 

_output_shapes
:W
Х
p
$__inference__update_step_xla_7097011
gradient

gradient_1	

gradient_2
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:€€€€€€€€€:€€€€€€€€€:: *
	_noinline(:Q M
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
gradient:MI
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
gradient:D@

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
ѓ
<
__inference__creator_7098009
identityИҐ
hash_tableo

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	6434507*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: /
NoOpNoOp^hash_table*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
і
В
.__inference_embedding_76_layer_call_fn_7097140

inputs	
unknown:
identityИҐStatefulPartitionedCallЎ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_embedding_76_layer_call_and_return_conditional_losses_7095237s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:€€€€€€€€€: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:'#
!
_user_specified_name	7097136
њ
c
G__inference_flatten_79_layer_call_and_return_conditional_losses_7097372

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:€€€€€€€€€X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
І
T
(__inference_dot_12_layer_call_fn_7097685
inputs_0
inputs_1
identityЊ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dot_12_layer_call_and_return_conditional_losses_7095552`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€ђ:€€€€€€€€€ђ:R N
(
_output_shapes
:€€€€€€€€€ђ
"
_user_specified_name
inputs_0:RN
(
_output_shapes
:€€€€€€€€€ђ
"
_user_specified_name
inputs_1
ѓ
Е
 __inference__initializer_7098076:
6key_value_init6434710_lookuptableimportv2_table_handle2
.key_value_init6434710_lookuptableimportv2_keys4
0key_value_init6434710_lookuptableimportv2_values	
identityИҐ)key_value_init6434710/LookupTableImportV2З
)key_value_init6434710/LookupTableImportV2LookupTableImportV26key_value_init6434710_lookuptableimportv2_table_handle.key_value_init6434710_lookuptableimportv2_keys0key_value_init6434710_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: N
NoOpNoOp*^key_value_init6434710/LookupTableImportV2*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2V
)key_value_init6434710/LookupTableImportV2)key_value_init6434710/LookupTableImportV2:, (
&
_user_specified_nametable_handle: 

_output_shapes
:: 

_output_shapes
:
Ѓ
H
,__inference_flatten_82_layer_call_fn_7097399

inputs
identityµ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_flatten_82_layer_call_and_return_conditional_losses_7095351`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
µ&
р
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_7097788

inputs6
'assignmovingavg_readvariableop_resource:	й8
)assignmovingavg_1_readvariableop_resource:	й4
%batchnorm_mul_readvariableop_resource:	й0
!batchnorm_readvariableop_resource:	й
identityИҐAssignMovingAvgҐAssignMovingAvg/ReadVariableOpҐAssignMovingAvg_1Ґ AssignMovingAvg_1/ReadVariableOpҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: А
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	й*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	йИ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:€€€€€€€€€йl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Я
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	й*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:й*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:й*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<Г
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:й*
dtype0В
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:йy
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:йђ
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<З
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:й*
dtype0И
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:й
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:йі
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:йQ
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:й
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:й*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:йd
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:€€€€€€€€€йi
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:йw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:й*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:йs
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:€€€€€€€€€йc
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€й∆
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€й: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€й
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
њ
c
G__inference_flatten_90_layer_call_and_return_conditional_losses_7095275

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€	   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:€€€€€€€€€	X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€	:S O
+
_output_shapes
:€€€€€€€€€	
 
_user_specified_nameinputs
Нг
Н6
E__inference_model_12_layer_call_and_return_conditional_losses_7096086
bmi
	age_range
	allergens
allergy
calories
carbohydrates
clinical_gender
cultural_factor
cultural_restriction
current_daily_calories
current_working_status

day_number

embeddings
	ethnicity
fat	
fiber

height

life_style
marital_status
meal_type_y
next_bmi
nutrition_goal
place_of_meal_consumption	
price
projected_daily_calories
protein(
$social_situation_of_meal_consumption	
taste
time_of_meal_consumption

weight@
<string_lookup_107_none_lookup_lookuptablefindv2_table_handleA
=string_lookup_107_none_lookup_lookuptablefindv2_default_value	@
<string_lookup_106_none_lookup_lookuptablefindv2_table_handleA
=string_lookup_106_none_lookup_lookuptablefindv2_default_value	@
<string_lookup_102_none_lookup_lookuptablefindv2_table_handleA
=string_lookup_102_none_lookup_lookuptablefindv2_default_value	@
<string_lookup_101_none_lookup_lookuptablefindv2_table_handleA
=string_lookup_101_none_lookup_lookuptablefindv2_default_value	@
<string_lookup_100_none_lookup_lookuptablefindv2_table_handleA
=string_lookup_100_none_lookup_lookuptablefindv2_default_value	?
;string_lookup_99_none_lookup_lookuptablefindv2_table_handle@
<string_lookup_99_none_lookup_lookuptablefindv2_default_value	?
;string_lookup_98_none_lookup_lookuptablefindv2_table_handle@
<string_lookup_98_none_lookup_lookuptablefindv2_default_value	?
;string_lookup_97_none_lookup_lookuptablefindv2_table_handle@
<string_lookup_97_none_lookup_lookuptablefindv2_default_value	?
;string_lookup_96_none_lookup_lookuptablefindv2_table_handle@
<string_lookup_96_none_lookup_lookuptablefindv2_default_value	?
;string_lookup_95_none_lookup_lookuptablefindv2_table_handle@
<string_lookup_95_none_lookup_lookuptablefindv2_default_value	?
;string_lookup_94_none_lookup_lookuptablefindv2_table_handle@
<string_lookup_94_none_lookup_lookuptablefindv2_default_value	?
;string_lookup_93_none_lookup_lookuptablefindv2_table_handle@
<string_lookup_93_none_lookup_lookuptablefindv2_default_value	?
;string_lookup_92_none_lookup_lookuptablefindv2_table_handle@
<string_lookup_92_none_lookup_lookuptablefindv2_default_value	@
<string_lookup_105_none_lookup_lookuptablefindv2_table_handleA
=string_lookup_105_none_lookup_lookuptablefindv2_default_value	@
<string_lookup_104_none_lookup_lookuptablefindv2_table_handleA
=string_lookup_104_none_lookup_lookuptablefindv2_default_value	&
embedding_90_7095894:W	&
embedding_89_7095897:&
embedding_86_7095900:&
embedding_85_7095903:&
embedding_84_7095906:&
embedding_83_7095909:&
embedding_82_7095912:&
embedding_81_7095915:&
embedding_80_7095918:&
embedding_79_7095921:&
embedding_78_7095924:&
embedding_77_7095927:&
embedding_76_7095930:&
embedding_88_7095933:&
embedding_87_7095936:@
<string_lookup_108_none_lookup_lookuptablefindv2_table_handleA
=string_lookup_108_none_lookup_lookuptablefindv2_default_value	@
<string_lookup_103_none_lookup_lookuptablefindv2_table_handleA
=string_lookup_103_none_lookup_lookuptablefindv2_default_value	)
user_embedding_7096010:	ђ%
user_embedding_7096012:	ђ*
food_embedding_7096015:
£ђ%
food_embedding_7096017:	ђ,
context_embedding_7096021:	Ш'
context_embedding_7096023:-
batch_normalization_12_7096028:	й-
batch_normalization_12_7096030:	й-
batch_normalization_12_7096032:	й-
batch_normalization_12_7096034:	й&
fc_layer_0_7096037:
йА!
fc_layer_0_7096039:	А%
fc_layer_1_7096042:	А@ 
fc_layer_1_7096044:@$
fc_layer_2_7096047:@  
fc_layer_2_7096049: "
output_0_7096052: 
output_0_7096054:
identityИҐ.batch_normalization_12/StatefulPartitionedCallҐ)context_embedding/StatefulPartitionedCallҐ:context_embedding/kernel/Regularizer/L2Loss/ReadVariableOpҐ$embedding_76/StatefulPartitionedCallҐ$embedding_77/StatefulPartitionedCallҐ$embedding_78/StatefulPartitionedCallҐ$embedding_79/StatefulPartitionedCallҐ$embedding_80/StatefulPartitionedCallҐ$embedding_81/StatefulPartitionedCallҐ$embedding_82/StatefulPartitionedCallҐ$embedding_83/StatefulPartitionedCallҐ$embedding_84/StatefulPartitionedCallҐ$embedding_85/StatefulPartitionedCallҐ$embedding_86/StatefulPartitionedCallҐ$embedding_87/StatefulPartitionedCallҐ$embedding_88/StatefulPartitionedCallҐ$embedding_89/StatefulPartitionedCallҐ$embedding_90/StatefulPartitionedCallҐ"fc_layer_0/StatefulPartitionedCallҐ3fc_layer_0/kernel/Regularizer/L2Loss/ReadVariableOpҐ"fc_layer_1/StatefulPartitionedCallҐ3fc_layer_1/kernel/Regularizer/L2Loss/ReadVariableOpҐ"fc_layer_2/StatefulPartitionedCallҐ3fc_layer_2/kernel/Regularizer/L2Loss/ReadVariableOpҐ&food_embedding/StatefulPartitionedCallҐ7food_embedding/kernel/Regularizer/L2Loss/ReadVariableOpҐ output_0/StatefulPartitionedCallҐ1output_0/kernel/Regularizer/L2Loss/ReadVariableOpҐstring_lookup_100/Assert/AssertҐ/string_lookup_100/None_Lookup/LookupTableFindV2Ґstring_lookup_101/Assert/AssertҐ/string_lookup_101/None_Lookup/LookupTableFindV2Ґstring_lookup_102/Assert/AssertҐ/string_lookup_102/None_Lookup/LookupTableFindV2Ґstring_lookup_103/Assert/AssertҐ/string_lookup_103/None_Lookup/LookupTableFindV2Ґstring_lookup_104/Assert/AssertҐ/string_lookup_104/None_Lookup/LookupTableFindV2Ґstring_lookup_105/Assert/AssertҐ/string_lookup_105/None_Lookup/LookupTableFindV2Ґstring_lookup_106/Assert/AssertҐ/string_lookup_106/None_Lookup/LookupTableFindV2Ґstring_lookup_107/Assert/AssertҐ/string_lookup_107/None_Lookup/LookupTableFindV2Ґstring_lookup_108/Assert/AssertҐ/string_lookup_108/None_Lookup/LookupTableFindV2Ґstring_lookup_92/Assert/AssertҐ.string_lookup_92/None_Lookup/LookupTableFindV2Ґstring_lookup_93/Assert/AssertҐ.string_lookup_93/None_Lookup/LookupTableFindV2Ґstring_lookup_94/Assert/AssertҐ.string_lookup_94/None_Lookup/LookupTableFindV2Ґstring_lookup_95/Assert/AssertҐ.string_lookup_95/None_Lookup/LookupTableFindV2Ґstring_lookup_96/Assert/AssertҐ.string_lookup_96/None_Lookup/LookupTableFindV2Ґstring_lookup_97/Assert/AssertҐ.string_lookup_97/None_Lookup/LookupTableFindV2Ґstring_lookup_98/Assert/AssertҐ.string_lookup_98/None_Lookup/LookupTableFindV2Ґstring_lookup_99/Assert/AssertҐ.string_lookup_99/None_Lookup/LookupTableFindV2Ґ&user_embedding/StatefulPartitionedCallҐ7user_embedding/kernel/Regularizer/L2Loss/ReadVariableOpМ
/string_lookup_107/None_Lookup/LookupTableFindV2LookupTableFindV2<string_lookup_107_none_lookup_lookuptablefindv2_table_handle	allergens=string_lookup_107_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:€€€€€€€€€d
string_lookup_107/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€∞
string_lookup_107/EqualEqual8string_lookup_107/None_Lookup/LookupTableFindV2:values:0"string_lookup_107/Equal/y:output:0*
T0	*'
_output_shapes
:€€€€€€€€€f
string_lookup_107/WhereWherestring_lookup_107/Equal:z:0*'
_output_shapes
:€€€€€€€€€Ц
string_lookup_107/GatherNdGatherNd	allergensstring_lookup_107/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:€€€€€€€€€Ф
string_lookup_107/StringFormatStringFormat#string_lookup_107/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*Е
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.`
string_lookup_107/SizeSizestring_lookup_107/Where:index:0*
T0	*
_output_shapes
: ]
string_lookup_107/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : К
string_lookup_107/Equal_1Equalstring_lookup_107/Size:output:0$string_lookup_107/Equal_1/y:output:0*
T0*
_output_shapes
: ї
string_lookup_107/Assert/AssertAssertstring_lookup_107/Equal_1:z:0'string_lookup_107/StringFormat:output:0*

T
2*&
 _has_manual_control_dependencies(*
_output_shapes
 і
string_lookup_107/IdentityIdentity8string_lookup_107/None_Lookup/LookupTableFindV2:values:0 ^string_lookup_107/Assert/Assert*
T0	*'
_output_shapes
:€€€€€€€€€Ч
/string_lookup_106/None_Lookup/LookupTableFindV2LookupTableFindV2<string_lookup_106_none_lookup_lookuptablefindv2_table_handlecultural_restriction=string_lookup_106_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:€€€€€€€€€d
string_lookup_106/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€∞
string_lookup_106/EqualEqual8string_lookup_106/None_Lookup/LookupTableFindV2:values:0"string_lookup_106/Equal/y:output:0*
T0	*'
_output_shapes
:€€€€€€€€€f
string_lookup_106/WhereWherestring_lookup_106/Equal:z:0*'
_output_shapes
:€€€€€€€€€°
string_lookup_106/GatherNdGatherNdcultural_restrictionstring_lookup_106/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:€€€€€€€€€Ф
string_lookup_106/StringFormatStringFormat#string_lookup_106/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*Е
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.`
string_lookup_106/SizeSizestring_lookup_106/Where:index:0*
T0	*
_output_shapes
: ]
string_lookup_106/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : К
string_lookup_106/Equal_1Equalstring_lookup_106/Size:output:0$string_lookup_106/Equal_1/y:output:0*
T0*
_output_shapes
: Ё
string_lookup_106/Assert/AssertAssertstring_lookup_106/Equal_1:z:0'string_lookup_106/StringFormat:output:0 ^string_lookup_107/Assert/Assert*

T
2*&
 _has_manual_control_dependencies(*
_output_shapes
 і
string_lookup_106/IdentityIdentity8string_lookup_106/None_Lookup/LookupTableFindV2:values:0 ^string_lookup_106/Assert/Assert*
T0	*'
_output_shapes
:€€€€€€€€€Л
/string_lookup_102/None_Lookup/LookupTableFindV2LookupTableFindV2<string_lookup_102_none_lookup_lookuptablefindv2_table_handlenext_bmi=string_lookup_102_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:€€€€€€€€€d
string_lookup_102/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€∞
string_lookup_102/EqualEqual8string_lookup_102/None_Lookup/LookupTableFindV2:values:0"string_lookup_102/Equal/y:output:0*
T0	*'
_output_shapes
:€€€€€€€€€f
string_lookup_102/WhereWherestring_lookup_102/Equal:z:0*'
_output_shapes
:€€€€€€€€€Х
string_lookup_102/GatherNdGatherNdnext_bmistring_lookup_102/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:€€€€€€€€€Ф
string_lookup_102/StringFormatStringFormat#string_lookup_102/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*Е
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.`
string_lookup_102/SizeSizestring_lookup_102/Where:index:0*
T0	*
_output_shapes
: ]
string_lookup_102/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : К
string_lookup_102/Equal_1Equalstring_lookup_102/Size:output:0$string_lookup_102/Equal_1/y:output:0*
T0*
_output_shapes
: Ё
string_lookup_102/Assert/AssertAssertstring_lookup_102/Equal_1:z:0'string_lookup_102/StringFormat:output:0 ^string_lookup_106/Assert/Assert*

T
2*&
 _has_manual_control_dependencies(*
_output_shapes
 і
string_lookup_102/IdentityIdentity8string_lookup_102/None_Lookup/LookupTableFindV2:values:0 ^string_lookup_102/Assert/Assert*
T0	*'
_output_shapes
:€€€€€€€€€Ж
/string_lookup_101/None_Lookup/LookupTableFindV2LookupTableFindV2<string_lookup_101_none_lookup_lookuptablefindv2_table_handlebmi=string_lookup_101_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:€€€€€€€€€d
string_lookup_101/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€∞
string_lookup_101/EqualEqual8string_lookup_101/None_Lookup/LookupTableFindV2:values:0"string_lookup_101/Equal/y:output:0*
T0	*'
_output_shapes
:€€€€€€€€€f
string_lookup_101/WhereWherestring_lookup_101/Equal:z:0*'
_output_shapes
:€€€€€€€€€Р
string_lookup_101/GatherNdGatherNdbmistring_lookup_101/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:€€€€€€€€€Ф
string_lookup_101/StringFormatStringFormat#string_lookup_101/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*Е
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.`
string_lookup_101/SizeSizestring_lookup_101/Where:index:0*
T0	*
_output_shapes
: ]
string_lookup_101/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : К
string_lookup_101/Equal_1Equalstring_lookup_101/Size:output:0$string_lookup_101/Equal_1/y:output:0*
T0*
_output_shapes
: Ё
string_lookup_101/Assert/AssertAssertstring_lookup_101/Equal_1:z:0'string_lookup_101/StringFormat:output:0 ^string_lookup_102/Assert/Assert*

T
2*&
 _has_manual_control_dependencies(*
_output_shapes
 і
string_lookup_101/IdentityIdentity8string_lookup_101/None_Lookup/LookupTableFindV2:values:0 ^string_lookup_101/Assert/Assert*
T0	*'
_output_shapes
:€€€€€€€€€М
/string_lookup_100/None_Lookup/LookupTableFindV2LookupTableFindV2<string_lookup_100_none_lookup_lookuptablefindv2_table_handle	ethnicity=string_lookup_100_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:€€€€€€€€€d
string_lookup_100/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€∞
string_lookup_100/EqualEqual8string_lookup_100/None_Lookup/LookupTableFindV2:values:0"string_lookup_100/Equal/y:output:0*
T0	*'
_output_shapes
:€€€€€€€€€f
string_lookup_100/WhereWherestring_lookup_100/Equal:z:0*'
_output_shapes
:€€€€€€€€€Ц
string_lookup_100/GatherNdGatherNd	ethnicitystring_lookup_100/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:€€€€€€€€€Ф
string_lookup_100/StringFormatStringFormat#string_lookup_100/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*Е
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.`
string_lookup_100/SizeSizestring_lookup_100/Where:index:0*
T0	*
_output_shapes
: ]
string_lookup_100/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : К
string_lookup_100/Equal_1Equalstring_lookup_100/Size:output:0$string_lookup_100/Equal_1/y:output:0*
T0*
_output_shapes
: Ё
string_lookup_100/Assert/AssertAssertstring_lookup_100/Equal_1:z:0'string_lookup_100/StringFormat:output:0 ^string_lookup_101/Assert/Assert*

T
2*&
 _has_manual_control_dependencies(*
_output_shapes
 і
string_lookup_100/IdentityIdentity8string_lookup_100/None_Lookup/LookupTableFindV2:values:0 ^string_lookup_100/Assert/Assert*
T0	*'
_output_shapes
:€€€€€€€€€О
.string_lookup_99/None_Lookup/LookupTableFindV2LookupTableFindV2;string_lookup_99_none_lookup_lookuptablefindv2_table_handlemarital_status<string_lookup_99_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:€€€€€€€€€c
string_lookup_99/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€≠
string_lookup_99/EqualEqual7string_lookup_99/None_Lookup/LookupTableFindV2:values:0!string_lookup_99/Equal/y:output:0*
T0	*'
_output_shapes
:€€€€€€€€€d
string_lookup_99/WhereWherestring_lookup_99/Equal:z:0*'
_output_shapes
:€€€€€€€€€Щ
string_lookup_99/GatherNdGatherNdmarital_statusstring_lookup_99/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:€€€€€€€€€Т
string_lookup_99/StringFormatStringFormat"string_lookup_99/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*Е
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.^
string_lookup_99/SizeSizestring_lookup_99/Where:index:0*
T0	*
_output_shapes
: \
string_lookup_99/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : З
string_lookup_99/Equal_1Equalstring_lookup_99/Size:output:0#string_lookup_99/Equal_1/y:output:0*
T0*
_output_shapes
: Џ
string_lookup_99/Assert/AssertAssertstring_lookup_99/Equal_1:z:0&string_lookup_99/StringFormat:output:0 ^string_lookup_100/Assert/Assert*

T
2*&
 _has_manual_control_dependencies(*
_output_shapes
 ±
string_lookup_99/IdentityIdentity7string_lookup_99/None_Lookup/LookupTableFindV2:values:0^string_lookup_99/Assert/Assert*
T0	*'
_output_shapes
:€€€€€€€€€Ц
.string_lookup_98/None_Lookup/LookupTableFindV2LookupTableFindV2;string_lookup_98_none_lookup_lookuptablefindv2_table_handlecurrent_working_status<string_lookup_98_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:€€€€€€€€€c
string_lookup_98/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€≠
string_lookup_98/EqualEqual7string_lookup_98/None_Lookup/LookupTableFindV2:values:0!string_lookup_98/Equal/y:output:0*
T0	*'
_output_shapes
:€€€€€€€€€d
string_lookup_98/WhereWherestring_lookup_98/Equal:z:0*'
_output_shapes
:€€€€€€€€€°
string_lookup_98/GatherNdGatherNdcurrent_working_statusstring_lookup_98/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:€€€€€€€€€Т
string_lookup_98/StringFormatStringFormat"string_lookup_98/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*Е
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.^
string_lookup_98/SizeSizestring_lookup_98/Where:index:0*
T0	*
_output_shapes
: \
string_lookup_98/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : З
string_lookup_98/Equal_1Equalstring_lookup_98/Size:output:0#string_lookup_98/Equal_1/y:output:0*
T0*
_output_shapes
: ў
string_lookup_98/Assert/AssertAssertstring_lookup_98/Equal_1:z:0&string_lookup_98/StringFormat:output:0^string_lookup_99/Assert/Assert*

T
2*&
 _has_manual_control_dependencies(*
_output_shapes
 ±
string_lookup_98/IdentityIdentity7string_lookup_98/None_Lookup/LookupTableFindV2:values:0^string_lookup_98/Assert/Assert*
T0	*'
_output_shapes
:€€€€€€€€€З
.string_lookup_97/None_Lookup/LookupTableFindV2LookupTableFindV2;string_lookup_97_none_lookup_lookuptablefindv2_table_handleallergy<string_lookup_97_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:€€€€€€€€€c
string_lookup_97/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€≠
string_lookup_97/EqualEqual7string_lookup_97/None_Lookup/LookupTableFindV2:values:0!string_lookup_97/Equal/y:output:0*
T0	*'
_output_shapes
:€€€€€€€€€d
string_lookup_97/WhereWherestring_lookup_97/Equal:z:0*'
_output_shapes
:€€€€€€€€€Т
string_lookup_97/GatherNdGatherNdallergystring_lookup_97/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:€€€€€€€€€Т
string_lookup_97/StringFormatStringFormat"string_lookup_97/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*Е
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.^
string_lookup_97/SizeSizestring_lookup_97/Where:index:0*
T0	*
_output_shapes
: \
string_lookup_97/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : З
string_lookup_97/Equal_1Equalstring_lookup_97/Size:output:0#string_lookup_97/Equal_1/y:output:0*
T0*
_output_shapes
: ў
string_lookup_97/Assert/AssertAssertstring_lookup_97/Equal_1:z:0&string_lookup_97/StringFormat:output:0^string_lookup_98/Assert/Assert*

T
2*&
 _has_manual_control_dependencies(*
_output_shapes
 ±
string_lookup_97/IdentityIdentity7string_lookup_97/None_Lookup/LookupTableFindV2:values:0^string_lookup_97/Assert/Assert*
T0	*'
_output_shapes
:€€€€€€€€€П
.string_lookup_96/None_Lookup/LookupTableFindV2LookupTableFindV2;string_lookup_96_none_lookup_lookuptablefindv2_table_handlecultural_factor<string_lookup_96_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:€€€€€€€€€c
string_lookup_96/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€≠
string_lookup_96/EqualEqual7string_lookup_96/None_Lookup/LookupTableFindV2:values:0!string_lookup_96/Equal/y:output:0*
T0	*'
_output_shapes
:€€€€€€€€€d
string_lookup_96/WhereWherestring_lookup_96/Equal:z:0*'
_output_shapes
:€€€€€€€€€Ъ
string_lookup_96/GatherNdGatherNdcultural_factorstring_lookup_96/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:€€€€€€€€€Т
string_lookup_96/StringFormatStringFormat"string_lookup_96/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*Е
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.^
string_lookup_96/SizeSizestring_lookup_96/Where:index:0*
T0	*
_output_shapes
: \
string_lookup_96/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : З
string_lookup_96/Equal_1Equalstring_lookup_96/Size:output:0#string_lookup_96/Equal_1/y:output:0*
T0*
_output_shapes
: ў
string_lookup_96/Assert/AssertAssertstring_lookup_96/Equal_1:z:0&string_lookup_96/StringFormat:output:0^string_lookup_97/Assert/Assert*

T
2*&
 _has_manual_control_dependencies(*
_output_shapes
 ±
string_lookup_96/IdentityIdentity7string_lookup_96/None_Lookup/LookupTableFindV2:values:0^string_lookup_96/Assert/Assert*
T0	*'
_output_shapes
:€€€€€€€€€К
.string_lookup_95/None_Lookup/LookupTableFindV2LookupTableFindV2;string_lookup_95_none_lookup_lookuptablefindv2_table_handle
life_style<string_lookup_95_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:€€€€€€€€€c
string_lookup_95/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€≠
string_lookup_95/EqualEqual7string_lookup_95/None_Lookup/LookupTableFindV2:values:0!string_lookup_95/Equal/y:output:0*
T0	*'
_output_shapes
:€€€€€€€€€d
string_lookup_95/WhereWherestring_lookup_95/Equal:z:0*'
_output_shapes
:€€€€€€€€€Х
string_lookup_95/GatherNdGatherNd
life_stylestring_lookup_95/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:€€€€€€€€€Т
string_lookup_95/StringFormatStringFormat"string_lookup_95/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*Е
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.^
string_lookup_95/SizeSizestring_lookup_95/Where:index:0*
T0	*
_output_shapes
: \
string_lookup_95/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : З
string_lookup_95/Equal_1Equalstring_lookup_95/Size:output:0#string_lookup_95/Equal_1/y:output:0*
T0*
_output_shapes
: ў
string_lookup_95/Assert/AssertAssertstring_lookup_95/Equal_1:z:0&string_lookup_95/StringFormat:output:0^string_lookup_96/Assert/Assert*

T
2*&
 _has_manual_control_dependencies(*
_output_shapes
 ±
string_lookup_95/IdentityIdentity7string_lookup_95/None_Lookup/LookupTableFindV2:values:0^string_lookup_95/Assert/Assert*
T0	*'
_output_shapes
:€€€€€€€€€Й
.string_lookup_94/None_Lookup/LookupTableFindV2LookupTableFindV2;string_lookup_94_none_lookup_lookuptablefindv2_table_handle	age_range<string_lookup_94_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:€€€€€€€€€c
string_lookup_94/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€≠
string_lookup_94/EqualEqual7string_lookup_94/None_Lookup/LookupTableFindV2:values:0!string_lookup_94/Equal/y:output:0*
T0	*'
_output_shapes
:€€€€€€€€€d
string_lookup_94/WhereWherestring_lookup_94/Equal:z:0*'
_output_shapes
:€€€€€€€€€Ф
string_lookup_94/GatherNdGatherNd	age_rangestring_lookup_94/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:€€€€€€€€€Т
string_lookup_94/StringFormatStringFormat"string_lookup_94/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*Е
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.^
string_lookup_94/SizeSizestring_lookup_94/Where:index:0*
T0	*
_output_shapes
: \
string_lookup_94/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : З
string_lookup_94/Equal_1Equalstring_lookup_94/Size:output:0#string_lookup_94/Equal_1/y:output:0*
T0*
_output_shapes
: ў
string_lookup_94/Assert/AssertAssertstring_lookup_94/Equal_1:z:0&string_lookup_94/StringFormat:output:0^string_lookup_95/Assert/Assert*

T
2*&
 _has_manual_control_dependencies(*
_output_shapes
 ±
string_lookup_94/IdentityIdentity7string_lookup_94/None_Lookup/LookupTableFindV2:values:0^string_lookup_94/Assert/Assert*
T0	*'
_output_shapes
:€€€€€€€€€П
.string_lookup_93/None_Lookup/LookupTableFindV2LookupTableFindV2;string_lookup_93_none_lookup_lookuptablefindv2_table_handleclinical_gender<string_lookup_93_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:€€€€€€€€€c
string_lookup_93/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€≠
string_lookup_93/EqualEqual7string_lookup_93/None_Lookup/LookupTableFindV2:values:0!string_lookup_93/Equal/y:output:0*
T0	*'
_output_shapes
:€€€€€€€€€d
string_lookup_93/WhereWherestring_lookup_93/Equal:z:0*'
_output_shapes
:€€€€€€€€€Ъ
string_lookup_93/GatherNdGatherNdclinical_genderstring_lookup_93/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:€€€€€€€€€Т
string_lookup_93/StringFormatStringFormat"string_lookup_93/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*Е
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.^
string_lookup_93/SizeSizestring_lookup_93/Where:index:0*
T0	*
_output_shapes
: \
string_lookup_93/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : З
string_lookup_93/Equal_1Equalstring_lookup_93/Size:output:0#string_lookup_93/Equal_1/y:output:0*
T0*
_output_shapes
: ў
string_lookup_93/Assert/AssertAssertstring_lookup_93/Equal_1:z:0&string_lookup_93/StringFormat:output:0^string_lookup_94/Assert/Assert*

T
2*&
 _has_manual_control_dependencies(*
_output_shapes
 ±
string_lookup_93/IdentityIdentity7string_lookup_93/None_Lookup/LookupTableFindV2:values:0^string_lookup_93/Assert/Assert*
T0	*'
_output_shapes
:€€€€€€€€€О
.string_lookup_92/None_Lookup/LookupTableFindV2LookupTableFindV2;string_lookup_92_none_lookup_lookuptablefindv2_table_handlenutrition_goal<string_lookup_92_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:€€€€€€€€€c
string_lookup_92/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€≠
string_lookup_92/EqualEqual7string_lookup_92/None_Lookup/LookupTableFindV2:values:0!string_lookup_92/Equal/y:output:0*
T0	*'
_output_shapes
:€€€€€€€€€d
string_lookup_92/WhereWherestring_lookup_92/Equal:z:0*'
_output_shapes
:€€€€€€€€€Щ
string_lookup_92/GatherNdGatherNdnutrition_goalstring_lookup_92/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:€€€€€€€€€Т
string_lookup_92/StringFormatStringFormat"string_lookup_92/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*Е
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.^
string_lookup_92/SizeSizestring_lookup_92/Where:index:0*
T0	*
_output_shapes
: \
string_lookup_92/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : З
string_lookup_92/Equal_1Equalstring_lookup_92/Size:output:0#string_lookup_92/Equal_1/y:output:0*
T0*
_output_shapes
: ў
string_lookup_92/Assert/AssertAssertstring_lookup_92/Equal_1:z:0&string_lookup_92/StringFormat:output:0^string_lookup_93/Assert/Assert*

T
2*&
 _has_manual_control_dependencies(*
_output_shapes
 ±
string_lookup_92/IdentityIdentity7string_lookup_92/None_Lookup/LookupTableFindV2:values:0^string_lookup_92/Assert/Assert*
T0	*'
_output_shapes
:€€€€€€€€€І
/string_lookup_105/None_Lookup/LookupTableFindV2LookupTableFindV2<string_lookup_105_none_lookup_lookuptablefindv2_table_handle$social_situation_of_meal_consumption=string_lookup_105_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:€€€€€€€€€d
string_lookup_105/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€∞
string_lookup_105/EqualEqual8string_lookup_105/None_Lookup/LookupTableFindV2:values:0"string_lookup_105/Equal/y:output:0*
T0	*'
_output_shapes
:€€€€€€€€€f
string_lookup_105/WhereWherestring_lookup_105/Equal:z:0*'
_output_shapes
:€€€€€€€€€±
string_lookup_105/GatherNdGatherNd$social_situation_of_meal_consumptionstring_lookup_105/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:€€€€€€€€€Ф
string_lookup_105/StringFormatStringFormat#string_lookup_105/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*Е
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.`
string_lookup_105/SizeSizestring_lookup_105/Where:index:0*
T0	*
_output_shapes
: ]
string_lookup_105/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : К
string_lookup_105/Equal_1Equalstring_lookup_105/Size:output:0$string_lookup_105/Equal_1/y:output:0*
T0*
_output_shapes
: №
string_lookup_105/Assert/AssertAssertstring_lookup_105/Equal_1:z:0'string_lookup_105/StringFormat:output:0^string_lookup_92/Assert/Assert*

T
2*&
 _has_manual_control_dependencies(*
_output_shapes
 і
string_lookup_105/IdentityIdentity8string_lookup_105/None_Lookup/LookupTableFindV2:values:0 ^string_lookup_105/Assert/Assert*
T0	*'
_output_shapes
:€€€€€€€€€Ь
/string_lookup_104/None_Lookup/LookupTableFindV2LookupTableFindV2<string_lookup_104_none_lookup_lookuptablefindv2_table_handleplace_of_meal_consumption=string_lookup_104_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:€€€€€€€€€d
string_lookup_104/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€∞
string_lookup_104/EqualEqual8string_lookup_104/None_Lookup/LookupTableFindV2:values:0"string_lookup_104/Equal/y:output:0*
T0	*'
_output_shapes
:€€€€€€€€€f
string_lookup_104/WhereWherestring_lookup_104/Equal:z:0*'
_output_shapes
:€€€€€€€€€¶
string_lookup_104/GatherNdGatherNdplace_of_meal_consumptionstring_lookup_104/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:€€€€€€€€€Ф
string_lookup_104/StringFormatStringFormat#string_lookup_104/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*Е
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.`
string_lookup_104/SizeSizestring_lookup_104/Where:index:0*
T0	*
_output_shapes
: ]
string_lookup_104/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : К
string_lookup_104/Equal_1Equalstring_lookup_104/Size:output:0$string_lookup_104/Equal_1/y:output:0*
T0*
_output_shapes
: Ё
string_lookup_104/Assert/AssertAssertstring_lookup_104/Equal_1:z:0'string_lookup_104/StringFormat:output:0 ^string_lookup_105/Assert/Assert*

T
2*&
 _has_manual_control_dependencies(*
_output_shapes
 і
string_lookup_104/IdentityIdentity8string_lookup_104/None_Lookup/LookupTableFindV2:values:0 ^string_lookup_104/Assert/Assert*
T0	*'
_output_shapes
:€€€€€€€€€П
$embedding_90/StatefulPartitionedCallStatefulPartitionedCall#string_lookup_107/Identity:output:0embedding_90_7095894*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€	*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_embedding_90_layer_call_and_return_conditional_losses_7095105П
$embedding_89/StatefulPartitionedCallStatefulPartitionedCall#string_lookup_106/Identity:output:0embedding_89_7095897*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_embedding_89_layer_call_and_return_conditional_losses_7095116П
$embedding_86/StatefulPartitionedCallStatefulPartitionedCall#string_lookup_102/Identity:output:0embedding_86_7095900*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_embedding_86_layer_call_and_return_conditional_losses_7095127П
$embedding_85/StatefulPartitionedCallStatefulPartitionedCall#string_lookup_101/Identity:output:0embedding_85_7095903*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_embedding_85_layer_call_and_return_conditional_losses_7095138П
$embedding_84/StatefulPartitionedCallStatefulPartitionedCall#string_lookup_100/Identity:output:0embedding_84_7095906*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_embedding_84_layer_call_and_return_conditional_losses_7095149О
$embedding_83/StatefulPartitionedCallStatefulPartitionedCall"string_lookup_99/Identity:output:0embedding_83_7095909*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_embedding_83_layer_call_and_return_conditional_losses_7095160О
$embedding_82/StatefulPartitionedCallStatefulPartitionedCall"string_lookup_98/Identity:output:0embedding_82_7095912*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_embedding_82_layer_call_and_return_conditional_losses_7095171О
$embedding_81/StatefulPartitionedCallStatefulPartitionedCall"string_lookup_97/Identity:output:0embedding_81_7095915*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_embedding_81_layer_call_and_return_conditional_losses_7095182О
$embedding_80/StatefulPartitionedCallStatefulPartitionedCall"string_lookup_96/Identity:output:0embedding_80_7095918*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_embedding_80_layer_call_and_return_conditional_losses_7095193О
$embedding_79/StatefulPartitionedCallStatefulPartitionedCall"string_lookup_95/Identity:output:0embedding_79_7095921*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_embedding_79_layer_call_and_return_conditional_losses_7095204О
$embedding_78/StatefulPartitionedCallStatefulPartitionedCall"string_lookup_94/Identity:output:0embedding_78_7095924*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_embedding_78_layer_call_and_return_conditional_losses_7095215О
$embedding_77/StatefulPartitionedCallStatefulPartitionedCall"string_lookup_93/Identity:output:0embedding_77_7095927*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_embedding_77_layer_call_and_return_conditional_losses_7095226О
$embedding_76/StatefulPartitionedCallStatefulPartitionedCall"string_lookup_92/Identity:output:0embedding_76_7095930*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_embedding_76_layer_call_and_return_conditional_losses_7095237П
$embedding_88/StatefulPartitionedCallStatefulPartitionedCall#string_lookup_105/Identity:output:0embedding_88_7095933*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_embedding_88_layer_call_and_return_conditional_losses_7095248П
$embedding_87/StatefulPartitionedCallStatefulPartitionedCall#string_lookup_104/Identity:output:0embedding_87_7095936*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_embedding_87_layer_call_and_return_conditional_losses_7095259з
flatten_89/PartitionedCallPartitionedCall-embedding_89/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_flatten_89_layer_call_and_return_conditional_losses_7095268з
flatten_90/PartitionedCallPartitionedCall-embedding_90/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_flatten_90_layer_call_and_return_conditional_losses_7095275И
/string_lookup_108/None_Lookup/LookupTableFindV2LookupTableFindV2<string_lookup_108_none_lookup_lookuptablefindv2_table_handletaste=string_lookup_108_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:€€€€€€€€€d
string_lookup_108/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€∞
string_lookup_108/EqualEqual8string_lookup_108/None_Lookup/LookupTableFindV2:values:0"string_lookup_108/Equal/y:output:0*
T0	*'
_output_shapes
:€€€€€€€€€f
string_lookup_108/WhereWherestring_lookup_108/Equal:z:0*'
_output_shapes
:€€€€€€€€€Т
string_lookup_108/GatherNdGatherNdtastestring_lookup_108/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:€€€€€€€€€Ф
string_lookup_108/StringFormatStringFormat#string_lookup_108/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*Е
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.`
string_lookup_108/SizeSizestring_lookup_108/Where:index:0*
T0	*
_output_shapes
: ]
string_lookup_108/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : К
string_lookup_108/Equal_1Equalstring_lookup_108/Size:output:0$string_lookup_108/Equal_1/y:output:0*
T0*
_output_shapes
: Ё
string_lookup_108/Assert/AssertAssertstring_lookup_108/Equal_1:z:0'string_lookup_108/StringFormat:output:0 ^string_lookup_104/Assert/Assert*

T
2*&
 _has_manual_control_dependencies(*
_output_shapes
 і
string_lookup_108/IdentityIdentity8string_lookup_108/None_Lookup/LookupTableFindV2:values:0 ^string_lookup_108/Assert/Assert*
T0	*'
_output_shapes
:€€€€€€€€€m
string_lookup_108/bincount/SizeSize#string_lookup_108/Identity:output:0*
T0	*
_output_shapes
: f
$string_lookup_108/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : І
"string_lookup_108/bincount/GreaterGreater(string_lookup_108/bincount/Size:output:0-string_lookup_108/bincount/Greater/y:output:0*
T0*
_output_shapes
: 
string_lookup_108/bincount/CastCast&string_lookup_108/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: q
 string_lookup_108/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Ц
string_lookup_108/bincount/MaxMax#string_lookup_108/Identity:output:0)string_lookup_108/bincount/Const:output:0*
T0	*
_output_shapes
: b
 string_lookup_108/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 RЬ
string_lookup_108/bincount/addAddV2'string_lookup_108/bincount/Max:output:0)string_lookup_108/bincount/add/y:output:0*
T0	*
_output_shapes
: П
string_lookup_108/bincount/mulMul#string_lookup_108/bincount/Cast:y:0"string_lookup_108/bincount/add:z:0*
T0	*
_output_shapes
: f
$string_lookup_108/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R°
"string_lookup_108/bincount/MaximumMaximum-string_lookup_108/bincount/minlength:output:0"string_lookup_108/bincount/mul:z:0*
T0	*
_output_shapes
: f
$string_lookup_108/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R•
"string_lookup_108/bincount/MinimumMinimum-string_lookup_108/bincount/maxlength:output:0&string_lookup_108/bincount/Maximum:z:0*
T0	*
_output_shapes
: e
"string_lookup_108/bincount/Const_1Const*
_output_shapes
: *
dtype0*
valueB Ж
(string_lookup_108/bincount/DenseBincountDenseBincount#string_lookup_108/Identity:output:0&string_lookup_108/bincount/Minimum:z:0+string_lookup_108/bincount/Const_1:output:0*
T0*

Tidx0	*'
_output_shapes
:€€€€€€€€€*
binary_output(з
flatten_76/PartitionedCallPartitionedCall-embedding_76/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_flatten_76_layer_call_and_return_conditional_losses_7095309з
flatten_77/PartitionedCallPartitionedCall-embedding_77/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_flatten_77_layer_call_and_return_conditional_losses_7095316з
flatten_78/PartitionedCallPartitionedCall-embedding_78/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_flatten_78_layer_call_and_return_conditional_losses_7095323з
flatten_79/PartitionedCallPartitionedCall-embedding_79/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_flatten_79_layer_call_and_return_conditional_losses_7095330з
flatten_80/PartitionedCallPartitionedCall-embedding_80/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_flatten_80_layer_call_and_return_conditional_losses_7095337з
flatten_81/PartitionedCallPartitionedCall-embedding_81/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_flatten_81_layer_call_and_return_conditional_losses_7095344з
flatten_82/PartitionedCallPartitionedCall-embedding_82/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_flatten_82_layer_call_and_return_conditional_losses_7095351з
flatten_83/PartitionedCallPartitionedCall-embedding_83/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_flatten_83_layer_call_and_return_conditional_losses_7095358з
flatten_84/PartitionedCallPartitionedCall-embedding_84/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_flatten_84_layer_call_and_return_conditional_losses_7095365з
flatten_85/PartitionedCallPartitionedCall-embedding_85/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_flatten_85_layer_call_and_return_conditional_losses_7095372з
flatten_86/PartitionedCallPartitionedCall-embedding_86/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_flatten_86_layer_call_and_return_conditional_losses_7095379О
/string_lookup_103/None_Lookup/LookupTableFindV2LookupTableFindV2<string_lookup_103_none_lookup_lookuptablefindv2_table_handlemeal_type_y=string_lookup_103_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:€€€€€€€€€d
string_lookup_103/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€∞
string_lookup_103/EqualEqual8string_lookup_103/None_Lookup/LookupTableFindV2:values:0"string_lookup_103/Equal/y:output:0*
T0	*'
_output_shapes
:€€€€€€€€€f
string_lookup_103/WhereWherestring_lookup_103/Equal:z:0*'
_output_shapes
:€€€€€€€€€Ш
string_lookup_103/GatherNdGatherNdmeal_type_ystring_lookup_103/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:€€€€€€€€€Ф
string_lookup_103/StringFormatStringFormat#string_lookup_103/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*Е
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.`
string_lookup_103/SizeSizestring_lookup_103/Where:index:0*
T0	*
_output_shapes
: ]
string_lookup_103/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : К
string_lookup_103/Equal_1Equalstring_lookup_103/Size:output:0$string_lookup_103/Equal_1/y:output:0*
T0*
_output_shapes
: Ё
string_lookup_103/Assert/AssertAssertstring_lookup_103/Equal_1:z:0'string_lookup_103/StringFormat:output:0 ^string_lookup_108/Assert/Assert*

T
2*&
 _has_manual_control_dependencies(*
_output_shapes
 і
string_lookup_103/IdentityIdentity8string_lookup_103/None_Lookup/LookupTableFindV2:values:0 ^string_lookup_103/Assert/Assert*
T0	*'
_output_shapes
:€€€€€€€€€m
string_lookup_103/bincount/SizeSize#string_lookup_103/Identity:output:0*
T0	*
_output_shapes
: f
$string_lookup_103/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : І
"string_lookup_103/bincount/GreaterGreater(string_lookup_103/bincount/Size:output:0-string_lookup_103/bincount/Greater/y:output:0*
T0*
_output_shapes
: 
string_lookup_103/bincount/CastCast&string_lookup_103/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: q
 string_lookup_103/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB"       Ц
string_lookup_103/bincount/MaxMax#string_lookup_103/Identity:output:0)string_lookup_103/bincount/Const:output:0*
T0	*
_output_shapes
: b
 string_lookup_103/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 RЬ
string_lookup_103/bincount/addAddV2'string_lookup_103/bincount/Max:output:0)string_lookup_103/bincount/add/y:output:0*
T0	*
_output_shapes
: П
string_lookup_103/bincount/mulMul#string_lookup_103/bincount/Cast:y:0"string_lookup_103/bincount/add:z:0*
T0	*
_output_shapes
: g
$string_lookup_103/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value
B	 RУ°
"string_lookup_103/bincount/MaximumMaximum-string_lookup_103/bincount/minlength:output:0"string_lookup_103/bincount/mul:z:0*
T0	*
_output_shapes
: g
$string_lookup_103/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
value
B	 RУ•
"string_lookup_103/bincount/MinimumMinimum-string_lookup_103/bincount/maxlength:output:0&string_lookup_103/bincount/Maximum:z:0*
T0	*
_output_shapes
: e
"string_lookup_103/bincount/Const_1Const*
_output_shapes
: *
dtype0*
valueB З
(string_lookup_103/bincount/DenseBincountDenseBincount#string_lookup_103/Identity:output:0&string_lookup_103/bincount/Minimum:z:0+string_lookup_103/bincount/Const_1:output:0*
T0*

Tidx0	*(
_output_shapes
:€€€€€€€€€У*
binary_output(з
flatten_87/PartitionedCallPartitionedCall-embedding_87/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_flatten_87_layer_call_and_return_conditional_losses_7095413з
flatten_88/PartitionedCallPartitionedCall-embedding_88/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_flatten_88_layer_call_and_return_conditional_losses_7095420И
concatenate_33/PartitionedCallPartitionedCall#flatten_89/PartitionedCall:output:0calories#flatten_90/PartitionedCall:output:01string_lookup_108/bincount/DenseBincount:output:0pricefiberfatproteincarbohydrates
embeddings*
Tin
2
*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€£* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_concatenate_33_layer_call_and_return_conditional_losses_7095436І
concatenate_31/PartitionedCallPartitionedCall#flatten_76/PartitionedCall:output:0#flatten_77/PartitionedCall:output:0#flatten_78/PartitionedCall:output:0#flatten_79/PartitionedCall:output:0weightheightprojected_daily_caloriescurrent_daily_calories#flatten_80/PartitionedCall:output:0#flatten_81/PartitionedCall:output:0#flatten_82/PartitionedCall:output:0#flatten_83/PartitionedCall:output:0#flatten_84/PartitionedCall:output:0#flatten_85/PartitionedCall:output:0#flatten_86/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_concatenate_31_layer_call_and_return_conditional_losses_7095457∞
&user_embedding/StatefulPartitionedCallStatefulPartitionedCall'concatenate_31/PartitionedCall:output:0user_embedding_7096010user_embedding_7096012*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_user_embedding_layer_call_and_return_conditional_losses_7095472∞
&food_embedding/StatefulPartitionedCallStatefulPartitionedCall'concatenate_33/PartitionedCall:output:0food_embedding_7096015food_embedding_7096017*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_food_embedding_layer_call_and_return_conditional_losses_7095491и
concatenate_32/PartitionedCallPartitionedCall
day_number1string_lookup_103/bincount/DenseBincount:output:0time_of_meal_consumption#flatten_87/PartitionedCall:output:0#flatten_88/PartitionedCall:output:0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€Ш* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_concatenate_32_layer_call_and_return_conditional_losses_7095506ї
)context_embedding/StatefulPartitionedCallStatefulPartitionedCall'concatenate_32/PartitionedCall:output:0context_embedding_7096021context_embedding_7096023*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_context_embedding_layer_call_and_return_conditional_losses_7095521У
dot_12/PartitionedCallPartitionedCall/user_embedding/StatefulPartitionedCall:output:0/food_embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dot_12_layer_call_and_return_conditional_losses_7095552ы
concatenate_34/PartitionedCallPartitionedCall/user_embedding/StatefulPartitionedCall:output:0/food_embedding/StatefulPartitionedCall:output:02context_embedding/StatefulPartitionedCall:output:0dot_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€й* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_concatenate_34_layer_call_and_return_conditional_losses_7095562Ф
.batch_normalization_12/StatefulPartitionedCallStatefulPartitionedCall'concatenate_34/PartitionedCall:output:0batch_normalization_12_7096028batch_normalization_12_7096030batch_normalization_12_7096032batch_normalization_12_7096034*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€й*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_7094860∞
"fc_layer_0/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_12/StatefulPartitionedCall:output:0fc_layer_0_7096037fc_layer_0_7096039*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_fc_layer_0_layer_call_and_return_conditional_losses_7095587£
"fc_layer_1/StatefulPartitionedCallStatefulPartitionedCall+fc_layer_0/StatefulPartitionedCall:output:0fc_layer_1_7096042fc_layer_1_7096044*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_fc_layer_1_layer_call_and_return_conditional_losses_7095607£
"fc_layer_2/StatefulPartitionedCallStatefulPartitionedCall+fc_layer_1/StatefulPartitionedCall:output:0fc_layer_2_7096047fc_layer_2_7096049*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_fc_layer_2_layer_call_and_return_conditional_losses_7095627Ы
 output_0/StatefulPartitionedCallStatefulPartitionedCall+fc_layer_2/StatefulPartitionedCall:output:0output_0_7096052output_0_7096054*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_output_0_layer_call_and_return_conditional_losses_7095647П
7user_embedding/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpuser_embedding_7096010*
_output_shapes
:	ђ*
dtype0Ф
(user_embedding/kernel/Regularizer/L2LossL2Loss?user_embedding/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: l
'user_embedding/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„£<≤
%user_embedding/kernel/Regularizer/mulMul0user_embedding/kernel/Regularizer/mul/x:output:01user_embedding/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Р
7food_embedding/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpfood_embedding_7096015* 
_output_shapes
:
£ђ*
dtype0Ф
(food_embedding/kernel/Regularizer/L2LossL2Loss?food_embedding/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: l
'food_embedding/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„£<≤
%food_embedding/kernel/Regularizer/mulMul0food_embedding/kernel/Regularizer/mul/x:output:01food_embedding/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Х
:context_embedding/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpcontext_embedding_7096021*
_output_shapes
:	Ш*
dtype0Ъ
+context_embedding/kernel/Regularizer/L2LossL2LossBcontext_embedding/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: o
*context_embedding/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„£<ї
(context_embedding/kernel/Regularizer/mulMul3context_embedding/kernel/Regularizer/mul/x:output:04context_embedding/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: И
3fc_layer_0/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpfc_layer_0_7096037* 
_output_shapes
:
йА*
dtype0М
$fc_layer_0/kernel/Regularizer/L2LossL2Loss;fc_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#fc_layer_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„£<¶
!fc_layer_0/kernel/Regularizer/mulMul,fc_layer_0/kernel/Regularizer/mul/x:output:0-fc_layer_0/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: З
3fc_layer_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpfc_layer_1_7096042*
_output_shapes
:	А@*
dtype0М
$fc_layer_1/kernel/Regularizer/L2LossL2Loss;fc_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#fc_layer_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„£<¶
!fc_layer_1/kernel/Regularizer/mulMul,fc_layer_1/kernel/Regularizer/mul/x:output:0-fc_layer_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Ж
3fc_layer_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpfc_layer_2_7096047*
_output_shapes

:@ *
dtype0М
$fc_layer_2/kernel/Regularizer/L2LossL2Loss;fc_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#fc_layer_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„£<¶
!fc_layer_2/kernel/Regularizer/mulMul,fc_layer_2/kernel/Regularizer/mul/x:output:0-fc_layer_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: В
1output_0/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpoutput_0_7096052*
_output_shapes

: *
dtype0И
"output_0/kernel/Regularizer/L2LossL2Loss9output_0/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!output_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„£<†
output_0/kernel/Regularizer/mulMul*output_0/kernel/Regularizer/mul/x:output:0+output_0/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: x
IdentityIdentity)output_0/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€Ј
NoOpNoOp/^batch_normalization_12/StatefulPartitionedCall*^context_embedding/StatefulPartitionedCall;^context_embedding/kernel/Regularizer/L2Loss/ReadVariableOp%^embedding_76/StatefulPartitionedCall%^embedding_77/StatefulPartitionedCall%^embedding_78/StatefulPartitionedCall%^embedding_79/StatefulPartitionedCall%^embedding_80/StatefulPartitionedCall%^embedding_81/StatefulPartitionedCall%^embedding_82/StatefulPartitionedCall%^embedding_83/StatefulPartitionedCall%^embedding_84/StatefulPartitionedCall%^embedding_85/StatefulPartitionedCall%^embedding_86/StatefulPartitionedCall%^embedding_87/StatefulPartitionedCall%^embedding_88/StatefulPartitionedCall%^embedding_89/StatefulPartitionedCall%^embedding_90/StatefulPartitionedCall#^fc_layer_0/StatefulPartitionedCall4^fc_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp#^fc_layer_1/StatefulPartitionedCall4^fc_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp#^fc_layer_2/StatefulPartitionedCall4^fc_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp'^food_embedding/StatefulPartitionedCall8^food_embedding/kernel/Regularizer/L2Loss/ReadVariableOp!^output_0/StatefulPartitionedCall2^output_0/kernel/Regularizer/L2Loss/ReadVariableOp ^string_lookup_100/Assert/Assert0^string_lookup_100/None_Lookup/LookupTableFindV2 ^string_lookup_101/Assert/Assert0^string_lookup_101/None_Lookup/LookupTableFindV2 ^string_lookup_102/Assert/Assert0^string_lookup_102/None_Lookup/LookupTableFindV2 ^string_lookup_103/Assert/Assert0^string_lookup_103/None_Lookup/LookupTableFindV2 ^string_lookup_104/Assert/Assert0^string_lookup_104/None_Lookup/LookupTableFindV2 ^string_lookup_105/Assert/Assert0^string_lookup_105/None_Lookup/LookupTableFindV2 ^string_lookup_106/Assert/Assert0^string_lookup_106/None_Lookup/LookupTableFindV2 ^string_lookup_107/Assert/Assert0^string_lookup_107/None_Lookup/LookupTableFindV2 ^string_lookup_108/Assert/Assert0^string_lookup_108/None_Lookup/LookupTableFindV2^string_lookup_92/Assert/Assert/^string_lookup_92/None_Lookup/LookupTableFindV2^string_lookup_93/Assert/Assert/^string_lookup_93/None_Lookup/LookupTableFindV2^string_lookup_94/Assert/Assert/^string_lookup_94/None_Lookup/LookupTableFindV2^string_lookup_95/Assert/Assert/^string_lookup_95/None_Lookup/LookupTableFindV2^string_lookup_96/Assert/Assert/^string_lookup_96/None_Lookup/LookupTableFindV2^string_lookup_97/Assert/Assert/^string_lookup_97/None_Lookup/LookupTableFindV2^string_lookup_98/Assert/Assert/^string_lookup_98/None_Lookup/LookupTableFindV2^string_lookup_99/Assert/Assert/^string_lookup_99/None_Lookup/LookupTableFindV2'^user_embedding/StatefulPartitionedCall8^user_embedding/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*÷
_input_shapesƒ
Ѕ:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€А:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_12/StatefulPartitionedCall.batch_normalization_12/StatefulPartitionedCall2V
)context_embedding/StatefulPartitionedCall)context_embedding/StatefulPartitionedCall2x
:context_embedding/kernel/Regularizer/L2Loss/ReadVariableOp:context_embedding/kernel/Regularizer/L2Loss/ReadVariableOp2L
$embedding_76/StatefulPartitionedCall$embedding_76/StatefulPartitionedCall2L
$embedding_77/StatefulPartitionedCall$embedding_77/StatefulPartitionedCall2L
$embedding_78/StatefulPartitionedCall$embedding_78/StatefulPartitionedCall2L
$embedding_79/StatefulPartitionedCall$embedding_79/StatefulPartitionedCall2L
$embedding_80/StatefulPartitionedCall$embedding_80/StatefulPartitionedCall2L
$embedding_81/StatefulPartitionedCall$embedding_81/StatefulPartitionedCall2L
$embedding_82/StatefulPartitionedCall$embedding_82/StatefulPartitionedCall2L
$embedding_83/StatefulPartitionedCall$embedding_83/StatefulPartitionedCall2L
$embedding_84/StatefulPartitionedCall$embedding_84/StatefulPartitionedCall2L
$embedding_85/StatefulPartitionedCall$embedding_85/StatefulPartitionedCall2L
$embedding_86/StatefulPartitionedCall$embedding_86/StatefulPartitionedCall2L
$embedding_87/StatefulPartitionedCall$embedding_87/StatefulPartitionedCall2L
$embedding_88/StatefulPartitionedCall$embedding_88/StatefulPartitionedCall2L
$embedding_89/StatefulPartitionedCall$embedding_89/StatefulPartitionedCall2L
$embedding_90/StatefulPartitionedCall$embedding_90/StatefulPartitionedCall2H
"fc_layer_0/StatefulPartitionedCall"fc_layer_0/StatefulPartitionedCall2j
3fc_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp3fc_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp2H
"fc_layer_1/StatefulPartitionedCall"fc_layer_1/StatefulPartitionedCall2j
3fc_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp3fc_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp2H
"fc_layer_2/StatefulPartitionedCall"fc_layer_2/StatefulPartitionedCall2j
3fc_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp3fc_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp2P
&food_embedding/StatefulPartitionedCall&food_embedding/StatefulPartitionedCall2r
7food_embedding/kernel/Regularizer/L2Loss/ReadVariableOp7food_embedding/kernel/Regularizer/L2Loss/ReadVariableOp2D
 output_0/StatefulPartitionedCall output_0/StatefulPartitionedCall2f
1output_0/kernel/Regularizer/L2Loss/ReadVariableOp1output_0/kernel/Regularizer/L2Loss/ReadVariableOp2B
string_lookup_100/Assert/Assertstring_lookup_100/Assert/Assert2b
/string_lookup_100/None_Lookup/LookupTableFindV2/string_lookup_100/None_Lookup/LookupTableFindV22B
string_lookup_101/Assert/Assertstring_lookup_101/Assert/Assert2b
/string_lookup_101/None_Lookup/LookupTableFindV2/string_lookup_101/None_Lookup/LookupTableFindV22B
string_lookup_102/Assert/Assertstring_lookup_102/Assert/Assert2b
/string_lookup_102/None_Lookup/LookupTableFindV2/string_lookup_102/None_Lookup/LookupTableFindV22B
string_lookup_103/Assert/Assertstring_lookup_103/Assert/Assert2b
/string_lookup_103/None_Lookup/LookupTableFindV2/string_lookup_103/None_Lookup/LookupTableFindV22B
string_lookup_104/Assert/Assertstring_lookup_104/Assert/Assert2b
/string_lookup_104/None_Lookup/LookupTableFindV2/string_lookup_104/None_Lookup/LookupTableFindV22B
string_lookup_105/Assert/Assertstring_lookup_105/Assert/Assert2b
/string_lookup_105/None_Lookup/LookupTableFindV2/string_lookup_105/None_Lookup/LookupTableFindV22B
string_lookup_106/Assert/Assertstring_lookup_106/Assert/Assert2b
/string_lookup_106/None_Lookup/LookupTableFindV2/string_lookup_106/None_Lookup/LookupTableFindV22B
string_lookup_107/Assert/Assertstring_lookup_107/Assert/Assert2b
/string_lookup_107/None_Lookup/LookupTableFindV2/string_lookup_107/None_Lookup/LookupTableFindV22B
string_lookup_108/Assert/Assertstring_lookup_108/Assert/Assert2b
/string_lookup_108/None_Lookup/LookupTableFindV2/string_lookup_108/None_Lookup/LookupTableFindV22@
string_lookup_92/Assert/Assertstring_lookup_92/Assert/Assert2`
.string_lookup_92/None_Lookup/LookupTableFindV2.string_lookup_92/None_Lookup/LookupTableFindV22@
string_lookup_93/Assert/Assertstring_lookup_93/Assert/Assert2`
.string_lookup_93/None_Lookup/LookupTableFindV2.string_lookup_93/None_Lookup/LookupTableFindV22@
string_lookup_94/Assert/Assertstring_lookup_94/Assert/Assert2`
.string_lookup_94/None_Lookup/LookupTableFindV2.string_lookup_94/None_Lookup/LookupTableFindV22@
string_lookup_95/Assert/Assertstring_lookup_95/Assert/Assert2`
.string_lookup_95/None_Lookup/LookupTableFindV2.string_lookup_95/None_Lookup/LookupTableFindV22@
string_lookup_96/Assert/Assertstring_lookup_96/Assert/Assert2`
.string_lookup_96/None_Lookup/LookupTableFindV2.string_lookup_96/None_Lookup/LookupTableFindV22@
string_lookup_97/Assert/Assertstring_lookup_97/Assert/Assert2`
.string_lookup_97/None_Lookup/LookupTableFindV2.string_lookup_97/None_Lookup/LookupTableFindV22@
string_lookup_98/Assert/Assertstring_lookup_98/Assert/Assert2`
.string_lookup_98/None_Lookup/LookupTableFindV2.string_lookup_98/None_Lookup/LookupTableFindV22@
string_lookup_99/Assert/Assertstring_lookup_99/Assert/Assert2`
.string_lookup_99/None_Lookup/LookupTableFindV2.string_lookup_99/None_Lookup/LookupTableFindV22P
&user_embedding/StatefulPartitionedCall&user_embedding/StatefulPartitionedCall2r
7user_embedding/kernel/Regularizer/L2Loss/ReadVariableOp7user_embedding/kernel/Regularizer/L2Loss/ReadVariableOp:L H
'
_output_shapes
:€€€€€€€€€

_user_specified_nameBMI:RN
'
_output_shapes
:€€€€€€€€€
#
_user_specified_name	age_range:RN
'
_output_shapes
:€€€€€€€€€
#
_user_specified_name	allergens:PL
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	allergy:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
calories:VR
'
_output_shapes
:€€€€€€€€€
'
_user_specified_namecarbohydrates:XT
'
_output_shapes
:€€€€€€€€€
)
_user_specified_nameclinical_gender:XT
'
_output_shapes
:€€€€€€€€€
)
_user_specified_namecultural_factor:]Y
'
_output_shapes
:€€€€€€€€€
.
_user_specified_namecultural_restriction:_	[
'
_output_shapes
:€€€€€€€€€
0
_user_specified_namecurrent_daily_calories:_
[
'
_output_shapes
:€€€€€€€€€
0
_user_specified_namecurrent_working_status:SO
'
_output_shapes
:€€€€€€€€€
$
_user_specified_name
day_number:TP
(
_output_shapes
:€€€€€€€€€А
$
_user_specified_name
embeddings:RN
'
_output_shapes
:€€€€€€€€€
#
_user_specified_name	ethnicity:LH
'
_output_shapes
:€€€€€€€€€

_user_specified_namefat:NJ
'
_output_shapes
:€€€€€€€€€

_user_specified_namefiber:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameheight:SO
'
_output_shapes
:€€€€€€€€€
$
_user_specified_name
life_style:WS
'
_output_shapes
:€€€€€€€€€
(
_user_specified_namemarital_status:TP
'
_output_shapes
:€€€€€€€€€
%
_user_specified_namemeal_type_y:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
next_BMI:WS
'
_output_shapes
:€€€€€€€€€
(
_user_specified_namenutrition_goal:b^
'
_output_shapes
:€€€€€€€€€
3
_user_specified_nameplace_of_meal_consumption:NJ
'
_output_shapes
:€€€€€€€€€

_user_specified_nameprice:a]
'
_output_shapes
:€€€€€€€€€
2
_user_specified_nameprojected_daily_calories:PL
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	protein:mi
'
_output_shapes
:€€€€€€€€€
>
_user_specified_name&$social_situation_of_meal_consumption:NJ
'
_output_shapes
:€€€€€€€€€

_user_specified_nametaste:a]
'
_output_shapes
:€€€€€€€€€
2
_user_specified_nametime_of_meal_consumption:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameweight:,(
&
_user_specified_nametable_handle:

_output_shapes
: :, (
&
_user_specified_nametable_handle:!

_output_shapes
: :,"(
&
_user_specified_nametable_handle:#

_output_shapes
: :,$(
&
_user_specified_nametable_handle:%

_output_shapes
: :,&(
&
_user_specified_nametable_handle:'

_output_shapes
: :,((
&
_user_specified_nametable_handle:)

_output_shapes
: :,*(
&
_user_specified_nametable_handle:+

_output_shapes
: :,,(
&
_user_specified_nametable_handle:-

_output_shapes
: :,.(
&
_user_specified_nametable_handle:/

_output_shapes
: :,0(
&
_user_specified_nametable_handle:1

_output_shapes
: :,2(
&
_user_specified_nametable_handle:3

_output_shapes
: :,4(
&
_user_specified_nametable_handle:5

_output_shapes
: :,6(
&
_user_specified_nametable_handle:7

_output_shapes
: :,8(
&
_user_specified_nametable_handle:9

_output_shapes
: :,:(
&
_user_specified_nametable_handle:;

_output_shapes
: :'<#
!
_user_specified_name	7095894:'=#
!
_user_specified_name	7095897:'>#
!
_user_specified_name	7095900:'?#
!
_user_specified_name	7095903:'@#
!
_user_specified_name	7095906:'A#
!
_user_specified_name	7095909:'B#
!
_user_specified_name	7095912:'C#
!
_user_specified_name	7095915:'D#
!
_user_specified_name	7095918:'E#
!
_user_specified_name	7095921:'F#
!
_user_specified_name	7095924:'G#
!
_user_specified_name	7095927:'H#
!
_user_specified_name	7095930:'I#
!
_user_specified_name	7095933:'J#
!
_user_specified_name	7095936:,K(
&
_user_specified_nametable_handle:L

_output_shapes
: :,M(
&
_user_specified_nametable_handle:N

_output_shapes
: :'O#
!
_user_specified_name	7096010:'P#
!
_user_specified_name	7096012:'Q#
!
_user_specified_name	7096015:'R#
!
_user_specified_name	7096017:'S#
!
_user_specified_name	7096021:'T#
!
_user_specified_name	7096023:'U#
!
_user_specified_name	7096028:'V#
!
_user_specified_name	7096030:'W#
!
_user_specified_name	7096032:'X#
!
_user_specified_name	7096034:'Y#
!
_user_specified_name	7096037:'Z#
!
_user_specified_name	7096039:'[#
!
_user_specified_name	7096042:'\#
!
_user_specified_name	7096044:']#
!
_user_specified_name	7096047:'^#
!
_user_specified_name	7096049:'_#
!
_user_specified_name	7096052:'`#
!
_user_specified_name	7096054
њ
c
G__inference_flatten_81_layer_call_and_return_conditional_losses_7097394

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:€€€€€€€€€X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
і
В
.__inference_embedding_89_layer_call_fn_7097305

inputs	
unknown:
identityИҐStatefulPartitionedCallЎ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_embedding_89_layer_call_and_return_conditional_losses_7095116s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:€€€€€€€€€: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:'#
!
_user_specified_name	7097301
ѓ
<
__inference__creator_7098129
identityИҐ
hash_tableo

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	6435139*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: /
NoOpNoOp^hash_table*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
«	
Є
__inference_loss_fn_3_7097936P
<fc_layer_0_kernel_regularizer_l2loss_readvariableop_resource:
йА
identityИҐ3fc_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp≤
3fc_layer_0/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp<fc_layer_0_kernel_regularizer_l2loss_readvariableop_resource* 
_output_shapes
:
йА*
dtype0М
$fc_layer_0/kernel/Regularizer/L2LossL2Loss;fc_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#fc_layer_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„£<¶
!fc_layer_0/kernel/Regularizer/mulMul,fc_layer_0/kernel/Regularizer/mul/x:output:0-fc_layer_0/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: c
IdentityIdentity%fc_layer_0/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: X
NoOpNoOp4^fc_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2j
3fc_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp3fc_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp:( $
"
_user_specified_name
resource
Ј	
„
8__inference_batch_normalization_12_layer_call_fn_7097754

inputs
unknown:	й
	unknown_0:	й
	unknown_1:	й
	unknown_2:	й
identityИҐStatefulPartitionedCallЖ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€й*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_7094860p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€й<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€й: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€й
 
_user_specified_nameinputs:'#
!
_user_specified_name	7097744:'#
!
_user_specified_name	7097746:'#
!
_user_specified_name	7097748:'#
!
_user_specified_name	7097750
г
љ
N__inference_context_embedding_layer_call_and_return_conditional_losses_7097679

inputs1
matmul_readvariableop_resource:	Ш-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐ:context_embedding/kernel/Regularizer/L2Loss/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ъ
:context_embedding/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype0Ъ
+context_embedding/kernel/Regularizer/L2LossL2LossBcontext_embedding/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: o
*context_embedding/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„£<ї
(context_embedding/kernel/Regularizer/mulMul3context_embedding/kernel/Regularizer/mul/x:output:04context_embedding/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€Р
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp;^context_embedding/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€Ш: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2x
:context_embedding/kernel/Regularizer/L2Loss/ReadVariableOp:context_embedding/kernel/Regularizer/L2Loss/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€Ш
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
њ
o
C__inference_dot_12_layer_call_and_return_conditional_losses_7097711
inputs_0
inputs_1
identityZ
l2_normalize/SquareSquareinputs_0*
T0*(
_output_shapes
:€€€€€€€€€ђd
"l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :†
l2_normalize/SumSuml2_normalize/Square:y:0+l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
	keep_dims([
l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *ћЉМ+Н
l2_normalize/MaximumMaximuml2_normalize/Sum:output:0l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€g
l2_normalize/RsqrtRsqrtl2_normalize/Maximum:z:0*
T0*'
_output_shapes
:€€€€€€€€€h
l2_normalizeMulinputs_0l2_normalize/Rsqrt:y:0*
T0*(
_output_shapes
:€€€€€€€€€ђ\
l2_normalize_1/SquareSquareinputs_1*
T0*(
_output_shapes
:€€€€€€€€€ђf
$l2_normalize_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :¶
l2_normalize_1/SumSuml2_normalize_1/Square:y:0-l2_normalize_1/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
	keep_dims(]
l2_normalize_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *ћЉМ+У
l2_normalize_1/MaximumMaximuml2_normalize_1/Sum:output:0!l2_normalize_1/Maximum/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€k
l2_normalize_1/RsqrtRsqrtl2_normalize_1/Maximum:z:0*
T0*'
_output_shapes
:€€€€€€€€€l
l2_normalize_1Mulinputs_1l2_normalize_1/Rsqrt:y:0*
T0*(
_output_shapes
:€€€€€€€€€ђP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :z

ExpandDims
ExpandDimsl2_normalize:z:0ExpandDims/dim:output:0*
T0*,
_output_shapes
:€€€€€€€€€ђR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :А
ExpandDims_1
ExpandDimsl2_normalize_1:z:0ExpandDims_1/dim:output:0*
T0*,
_output_shapes
:€€€€€€€€€ђy
MatMulBatchMatMulV2ExpandDims:output:0ExpandDims_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€R
ShapeShapeMatMul:output:0*
T0*
_output_shapes
::нѕl
SqueezeSqueezeMatMul:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
squeeze_dims
X
IdentityIdentitySqueeze:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€ђ:€€€€€€€€€ђ:R N
(
_output_shapes
:€€€€€€€€€ђ
"
_user_specified_name
inputs_0:RN
(
_output_shapes
:€€€€€€€€€ђ
"
_user_specified_name
inputs_1
ѓ
<
__inference__creator_7098069
identityИҐ
hash_tableo

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	6434711*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: /
NoOpNoOp^hash_table*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
И
†
0__inference_food_embedding_layer_call_fn_7097623

inputs
unknown:
£ђ
	unknown_0:	ђ
identityИҐStatefulPartitionedCallд
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_food_embedding_layer_call_and_return_conditional_losses_7095491p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€ђ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€£: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€£
 
_user_specified_nameinputs:'#
!
_user_specified_name	7097617:'#
!
_user_specified_name	7097619
Ѓ
H
,__inference_flatten_88_layer_call_fn_7097585

inputs
identityµ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_flatten_88_layer_call_and_return_conditional_losses_7095420`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ѓ
H
,__inference_flatten_77_layer_call_fn_7097344

inputs
identityµ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_flatten_77_layer_call_and_return_conditional_losses_7095316`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Х
p
$__inference__update_step_xla_7097025
gradient

gradient_1	

gradient_2
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:€€€€€€€€€:€€€€€€€€€:: *
	_noinline(:Q M
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
gradient:MI
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
gradient:D@

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
√	
ґ
__inference_loss_fn_5_7097952N
<fc_layer_2_kernel_regularizer_l2loss_readvariableop_resource:@ 
identityИҐ3fc_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp∞
3fc_layer_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp<fc_layer_2_kernel_regularizer_l2loss_readvariableop_resource*
_output_shapes

:@ *
dtype0М
$fc_layer_2/kernel/Regularizer/L2LossL2Loss;fc_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#fc_layer_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„£<¶
!fc_layer_2/kernel/Regularizer/mulMul,fc_layer_2/kernel/Regularizer/mul/x:output:0-fc_layer_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: c
IdentityIdentity%fc_layer_2/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: X
NoOpNoOp4^fc_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2j
3fc_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp3fc_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp:( $
"
_user_specified_name
resource
і
В
.__inference_embedding_78_layer_call_fn_7097170

inputs	
unknown:
identityИҐStatefulPartitionedCallЎ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_embedding_78_layer_call_and_return_conditional_losses_7095215s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:€€€€€€€€€: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:'#
!
_user_specified_name	7097166
Ѓ
H
,__inference_flatten_79_layer_call_fn_7097366

inputs
identityµ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_flatten_79_layer_call_and_return_conditional_losses_7095330`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
‘≥
ВD
"__inference__wrapped_model_7094806
bmi
	age_range
	allergens
allergy
calories
carbohydrates
clinical_gender
cultural_factor
cultural_restriction
current_daily_calories
current_working_status

day_number

embeddings
	ethnicity
fat	
fiber

height

life_style
marital_status
meal_type_y
next_bmi
nutrition_goal
place_of_meal_consumption	
price
projected_daily_calories
protein(
$social_situation_of_meal_consumption	
taste
time_of_meal_consumption

weightI
Emodel_12_string_lookup_107_none_lookup_lookuptablefindv2_table_handleJ
Fmodel_12_string_lookup_107_none_lookup_lookuptablefindv2_default_value	I
Emodel_12_string_lookup_106_none_lookup_lookuptablefindv2_table_handleJ
Fmodel_12_string_lookup_106_none_lookup_lookuptablefindv2_default_value	I
Emodel_12_string_lookup_102_none_lookup_lookuptablefindv2_table_handleJ
Fmodel_12_string_lookup_102_none_lookup_lookuptablefindv2_default_value	I
Emodel_12_string_lookup_101_none_lookup_lookuptablefindv2_table_handleJ
Fmodel_12_string_lookup_101_none_lookup_lookuptablefindv2_default_value	I
Emodel_12_string_lookup_100_none_lookup_lookuptablefindv2_table_handleJ
Fmodel_12_string_lookup_100_none_lookup_lookuptablefindv2_default_value	H
Dmodel_12_string_lookup_99_none_lookup_lookuptablefindv2_table_handleI
Emodel_12_string_lookup_99_none_lookup_lookuptablefindv2_default_value	H
Dmodel_12_string_lookup_98_none_lookup_lookuptablefindv2_table_handleI
Emodel_12_string_lookup_98_none_lookup_lookuptablefindv2_default_value	H
Dmodel_12_string_lookup_97_none_lookup_lookuptablefindv2_table_handleI
Emodel_12_string_lookup_97_none_lookup_lookuptablefindv2_default_value	H
Dmodel_12_string_lookup_96_none_lookup_lookuptablefindv2_table_handleI
Emodel_12_string_lookup_96_none_lookup_lookuptablefindv2_default_value	H
Dmodel_12_string_lookup_95_none_lookup_lookuptablefindv2_table_handleI
Emodel_12_string_lookup_95_none_lookup_lookuptablefindv2_default_value	H
Dmodel_12_string_lookup_94_none_lookup_lookuptablefindv2_table_handleI
Emodel_12_string_lookup_94_none_lookup_lookuptablefindv2_default_value	H
Dmodel_12_string_lookup_93_none_lookup_lookuptablefindv2_table_handleI
Emodel_12_string_lookup_93_none_lookup_lookuptablefindv2_default_value	H
Dmodel_12_string_lookup_92_none_lookup_lookuptablefindv2_table_handleI
Emodel_12_string_lookup_92_none_lookup_lookuptablefindv2_default_value	I
Emodel_12_string_lookup_105_none_lookup_lookuptablefindv2_table_handleJ
Fmodel_12_string_lookup_105_none_lookup_lookuptablefindv2_default_value	I
Emodel_12_string_lookup_104_none_lookup_lookuptablefindv2_table_handleJ
Fmodel_12_string_lookup_104_none_lookup_lookuptablefindv2_default_value	@
.model_12_embedding_90_embedding_lookup_7094570:W	@
.model_12_embedding_89_embedding_lookup_7094574:@
.model_12_embedding_86_embedding_lookup_7094578:@
.model_12_embedding_85_embedding_lookup_7094582:@
.model_12_embedding_84_embedding_lookup_7094586:@
.model_12_embedding_83_embedding_lookup_7094590:@
.model_12_embedding_82_embedding_lookup_7094594:@
.model_12_embedding_81_embedding_lookup_7094598:@
.model_12_embedding_80_embedding_lookup_7094602:@
.model_12_embedding_79_embedding_lookup_7094606:@
.model_12_embedding_78_embedding_lookup_7094610:@
.model_12_embedding_77_embedding_lookup_7094614:@
.model_12_embedding_76_embedding_lookup_7094618:@
.model_12_embedding_88_embedding_lookup_7094622:@
.model_12_embedding_87_embedding_lookup_7094626:I
Emodel_12_string_lookup_108_none_lookup_lookuptablefindv2_table_handleJ
Fmodel_12_string_lookup_108_none_lookup_lookuptablefindv2_default_value	I
Emodel_12_string_lookup_103_none_lookup_lookuptablefindv2_table_handleJ
Fmodel_12_string_lookup_103_none_lookup_lookuptablefindv2_default_value	I
6model_12_user_embedding_matmul_readvariableop_resource:	ђF
7model_12_user_embedding_biasadd_readvariableop_resource:	ђJ
6model_12_food_embedding_matmul_readvariableop_resource:
£ђF
7model_12_food_embedding_biasadd_readvariableop_resource:	ђL
9model_12_context_embedding_matmul_readvariableop_resource:	ШH
:model_12_context_embedding_biasadd_readvariableop_resource:P
Amodel_12_batch_normalization_12_batchnorm_readvariableop_resource:	йT
Emodel_12_batch_normalization_12_batchnorm_mul_readvariableop_resource:	йR
Cmodel_12_batch_normalization_12_batchnorm_readvariableop_1_resource:	йR
Cmodel_12_batch_normalization_12_batchnorm_readvariableop_2_resource:	йF
2model_12_fc_layer_0_matmul_readvariableop_resource:
йАB
3model_12_fc_layer_0_biasadd_readvariableop_resource:	АE
2model_12_fc_layer_1_matmul_readvariableop_resource:	А@A
3model_12_fc_layer_1_biasadd_readvariableop_resource:@D
2model_12_fc_layer_2_matmul_readvariableop_resource:@ A
3model_12_fc_layer_2_biasadd_readvariableop_resource: B
0model_12_output_0_matmul_readvariableop_resource: ?
1model_12_output_0_biasadd_readvariableop_resource:
identityИҐ8model_12/batch_normalization_12/batchnorm/ReadVariableOpҐ:model_12/batch_normalization_12/batchnorm/ReadVariableOp_1Ґ:model_12/batch_normalization_12/batchnorm/ReadVariableOp_2Ґ<model_12/batch_normalization_12/batchnorm/mul/ReadVariableOpҐ1model_12/context_embedding/BiasAdd/ReadVariableOpҐ0model_12/context_embedding/MatMul/ReadVariableOpҐ&model_12/embedding_76/embedding_lookupҐ&model_12/embedding_77/embedding_lookupҐ&model_12/embedding_78/embedding_lookupҐ&model_12/embedding_79/embedding_lookupҐ&model_12/embedding_80/embedding_lookupҐ&model_12/embedding_81/embedding_lookupҐ&model_12/embedding_82/embedding_lookupҐ&model_12/embedding_83/embedding_lookupҐ&model_12/embedding_84/embedding_lookupҐ&model_12/embedding_85/embedding_lookupҐ&model_12/embedding_86/embedding_lookupҐ&model_12/embedding_87/embedding_lookupҐ&model_12/embedding_88/embedding_lookupҐ&model_12/embedding_89/embedding_lookupҐ&model_12/embedding_90/embedding_lookupҐ*model_12/fc_layer_0/BiasAdd/ReadVariableOpҐ)model_12/fc_layer_0/MatMul/ReadVariableOpҐ*model_12/fc_layer_1/BiasAdd/ReadVariableOpҐ)model_12/fc_layer_1/MatMul/ReadVariableOpҐ*model_12/fc_layer_2/BiasAdd/ReadVariableOpҐ)model_12/fc_layer_2/MatMul/ReadVariableOpҐ.model_12/food_embedding/BiasAdd/ReadVariableOpҐ-model_12/food_embedding/MatMul/ReadVariableOpҐ(model_12/output_0/BiasAdd/ReadVariableOpҐ'model_12/output_0/MatMul/ReadVariableOpҐ(model_12/string_lookup_100/Assert/AssertҐ8model_12/string_lookup_100/None_Lookup/LookupTableFindV2Ґ(model_12/string_lookup_101/Assert/AssertҐ8model_12/string_lookup_101/None_Lookup/LookupTableFindV2Ґ(model_12/string_lookup_102/Assert/AssertҐ8model_12/string_lookup_102/None_Lookup/LookupTableFindV2Ґ(model_12/string_lookup_103/Assert/AssertҐ8model_12/string_lookup_103/None_Lookup/LookupTableFindV2Ґ(model_12/string_lookup_104/Assert/AssertҐ8model_12/string_lookup_104/None_Lookup/LookupTableFindV2Ґ(model_12/string_lookup_105/Assert/AssertҐ8model_12/string_lookup_105/None_Lookup/LookupTableFindV2Ґ(model_12/string_lookup_106/Assert/AssertҐ8model_12/string_lookup_106/None_Lookup/LookupTableFindV2Ґ(model_12/string_lookup_107/Assert/AssertҐ8model_12/string_lookup_107/None_Lookup/LookupTableFindV2Ґ(model_12/string_lookup_108/Assert/AssertҐ8model_12/string_lookup_108/None_Lookup/LookupTableFindV2Ґ'model_12/string_lookup_92/Assert/AssertҐ7model_12/string_lookup_92/None_Lookup/LookupTableFindV2Ґ'model_12/string_lookup_93/Assert/AssertҐ7model_12/string_lookup_93/None_Lookup/LookupTableFindV2Ґ'model_12/string_lookup_94/Assert/AssertҐ7model_12/string_lookup_94/None_Lookup/LookupTableFindV2Ґ'model_12/string_lookup_95/Assert/AssertҐ7model_12/string_lookup_95/None_Lookup/LookupTableFindV2Ґ'model_12/string_lookup_96/Assert/AssertҐ7model_12/string_lookup_96/None_Lookup/LookupTableFindV2Ґ'model_12/string_lookup_97/Assert/AssertҐ7model_12/string_lookup_97/None_Lookup/LookupTableFindV2Ґ'model_12/string_lookup_98/Assert/AssertҐ7model_12/string_lookup_98/None_Lookup/LookupTableFindV2Ґ'model_12/string_lookup_99/Assert/AssertҐ7model_12/string_lookup_99/None_Lookup/LookupTableFindV2Ґ.model_12/user_embedding/BiasAdd/ReadVariableOpҐ-model_12/user_embedding/MatMul/ReadVariableOpІ
8model_12/string_lookup_107/None_Lookup/LookupTableFindV2LookupTableFindV2Emodel_12_string_lookup_107_none_lookup_lookuptablefindv2_table_handle	allergensFmodel_12_string_lookup_107_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:€€€€€€€€€m
"model_12/string_lookup_107/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€Ћ
 model_12/string_lookup_107/EqualEqualAmodel_12/string_lookup_107/None_Lookup/LookupTableFindV2:values:0+model_12/string_lookup_107/Equal/y:output:0*
T0	*'
_output_shapes
:€€€€€€€€€x
 model_12/string_lookup_107/WhereWhere$model_12/string_lookup_107/Equal:z:0*'
_output_shapes
:€€€€€€€€€®
#model_12/string_lookup_107/GatherNdGatherNd	allergens(model_12/string_lookup_107/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:€€€€€€€€€¶
'model_12/string_lookup_107/StringFormatStringFormat,model_12/string_lookup_107/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*Е
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.r
model_12/string_lookup_107/SizeSize(model_12/string_lookup_107/Where:index:0*
T0	*
_output_shapes
: f
$model_12/string_lookup_107/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : •
"model_12/string_lookup_107/Equal_1Equal(model_12/string_lookup_107/Size:output:0-model_12/string_lookup_107/Equal_1/y:output:0*
T0*
_output_shapes
: ÷
(model_12/string_lookup_107/Assert/AssertAssert&model_12/string_lookup_107/Equal_1:z:00model_12/string_lookup_107/StringFormat:output:0*

T
2*&
 _has_manual_control_dependencies(*
_output_shapes
 ѕ
#model_12/string_lookup_107/IdentityIdentityAmodel_12/string_lookup_107/None_Lookup/LookupTableFindV2:values:0)^model_12/string_lookup_107/Assert/Assert*
T0	*'
_output_shapes
:€€€€€€€€€≤
8model_12/string_lookup_106/None_Lookup/LookupTableFindV2LookupTableFindV2Emodel_12_string_lookup_106_none_lookup_lookuptablefindv2_table_handlecultural_restrictionFmodel_12_string_lookup_106_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:€€€€€€€€€m
"model_12/string_lookup_106/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€Ћ
 model_12/string_lookup_106/EqualEqualAmodel_12/string_lookup_106/None_Lookup/LookupTableFindV2:values:0+model_12/string_lookup_106/Equal/y:output:0*
T0	*'
_output_shapes
:€€€€€€€€€x
 model_12/string_lookup_106/WhereWhere$model_12/string_lookup_106/Equal:z:0*'
_output_shapes
:€€€€€€€€€≥
#model_12/string_lookup_106/GatherNdGatherNdcultural_restriction(model_12/string_lookup_106/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:€€€€€€€€€¶
'model_12/string_lookup_106/StringFormatStringFormat,model_12/string_lookup_106/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*Е
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.r
model_12/string_lookup_106/SizeSize(model_12/string_lookup_106/Where:index:0*
T0	*
_output_shapes
: f
$model_12/string_lookup_106/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : •
"model_12/string_lookup_106/Equal_1Equal(model_12/string_lookup_106/Size:output:0-model_12/string_lookup_106/Equal_1/y:output:0*
T0*
_output_shapes
: Б
(model_12/string_lookup_106/Assert/AssertAssert&model_12/string_lookup_106/Equal_1:z:00model_12/string_lookup_106/StringFormat:output:0)^model_12/string_lookup_107/Assert/Assert*

T
2*&
 _has_manual_control_dependencies(*
_output_shapes
 ѕ
#model_12/string_lookup_106/IdentityIdentityAmodel_12/string_lookup_106/None_Lookup/LookupTableFindV2:values:0)^model_12/string_lookup_106/Assert/Assert*
T0	*'
_output_shapes
:€€€€€€€€€¶
8model_12/string_lookup_102/None_Lookup/LookupTableFindV2LookupTableFindV2Emodel_12_string_lookup_102_none_lookup_lookuptablefindv2_table_handlenext_bmiFmodel_12_string_lookup_102_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:€€€€€€€€€m
"model_12/string_lookup_102/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€Ћ
 model_12/string_lookup_102/EqualEqualAmodel_12/string_lookup_102/None_Lookup/LookupTableFindV2:values:0+model_12/string_lookup_102/Equal/y:output:0*
T0	*'
_output_shapes
:€€€€€€€€€x
 model_12/string_lookup_102/WhereWhere$model_12/string_lookup_102/Equal:z:0*'
_output_shapes
:€€€€€€€€€І
#model_12/string_lookup_102/GatherNdGatherNdnext_bmi(model_12/string_lookup_102/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:€€€€€€€€€¶
'model_12/string_lookup_102/StringFormatStringFormat,model_12/string_lookup_102/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*Е
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.r
model_12/string_lookup_102/SizeSize(model_12/string_lookup_102/Where:index:0*
T0	*
_output_shapes
: f
$model_12/string_lookup_102/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : •
"model_12/string_lookup_102/Equal_1Equal(model_12/string_lookup_102/Size:output:0-model_12/string_lookup_102/Equal_1/y:output:0*
T0*
_output_shapes
: Б
(model_12/string_lookup_102/Assert/AssertAssert&model_12/string_lookup_102/Equal_1:z:00model_12/string_lookup_102/StringFormat:output:0)^model_12/string_lookup_106/Assert/Assert*

T
2*&
 _has_manual_control_dependencies(*
_output_shapes
 ѕ
#model_12/string_lookup_102/IdentityIdentityAmodel_12/string_lookup_102/None_Lookup/LookupTableFindV2:values:0)^model_12/string_lookup_102/Assert/Assert*
T0	*'
_output_shapes
:€€€€€€€€€°
8model_12/string_lookup_101/None_Lookup/LookupTableFindV2LookupTableFindV2Emodel_12_string_lookup_101_none_lookup_lookuptablefindv2_table_handlebmiFmodel_12_string_lookup_101_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:€€€€€€€€€m
"model_12/string_lookup_101/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€Ћ
 model_12/string_lookup_101/EqualEqualAmodel_12/string_lookup_101/None_Lookup/LookupTableFindV2:values:0+model_12/string_lookup_101/Equal/y:output:0*
T0	*'
_output_shapes
:€€€€€€€€€x
 model_12/string_lookup_101/WhereWhere$model_12/string_lookup_101/Equal:z:0*'
_output_shapes
:€€€€€€€€€Ґ
#model_12/string_lookup_101/GatherNdGatherNdbmi(model_12/string_lookup_101/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:€€€€€€€€€¶
'model_12/string_lookup_101/StringFormatStringFormat,model_12/string_lookup_101/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*Е
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.r
model_12/string_lookup_101/SizeSize(model_12/string_lookup_101/Where:index:0*
T0	*
_output_shapes
: f
$model_12/string_lookup_101/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : •
"model_12/string_lookup_101/Equal_1Equal(model_12/string_lookup_101/Size:output:0-model_12/string_lookup_101/Equal_1/y:output:0*
T0*
_output_shapes
: Б
(model_12/string_lookup_101/Assert/AssertAssert&model_12/string_lookup_101/Equal_1:z:00model_12/string_lookup_101/StringFormat:output:0)^model_12/string_lookup_102/Assert/Assert*

T
2*&
 _has_manual_control_dependencies(*
_output_shapes
 ѕ
#model_12/string_lookup_101/IdentityIdentityAmodel_12/string_lookup_101/None_Lookup/LookupTableFindV2:values:0)^model_12/string_lookup_101/Assert/Assert*
T0	*'
_output_shapes
:€€€€€€€€€І
8model_12/string_lookup_100/None_Lookup/LookupTableFindV2LookupTableFindV2Emodel_12_string_lookup_100_none_lookup_lookuptablefindv2_table_handle	ethnicityFmodel_12_string_lookup_100_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:€€€€€€€€€m
"model_12/string_lookup_100/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€Ћ
 model_12/string_lookup_100/EqualEqualAmodel_12/string_lookup_100/None_Lookup/LookupTableFindV2:values:0+model_12/string_lookup_100/Equal/y:output:0*
T0	*'
_output_shapes
:€€€€€€€€€x
 model_12/string_lookup_100/WhereWhere$model_12/string_lookup_100/Equal:z:0*'
_output_shapes
:€€€€€€€€€®
#model_12/string_lookup_100/GatherNdGatherNd	ethnicity(model_12/string_lookup_100/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:€€€€€€€€€¶
'model_12/string_lookup_100/StringFormatStringFormat,model_12/string_lookup_100/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*Е
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.r
model_12/string_lookup_100/SizeSize(model_12/string_lookup_100/Where:index:0*
T0	*
_output_shapes
: f
$model_12/string_lookup_100/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : •
"model_12/string_lookup_100/Equal_1Equal(model_12/string_lookup_100/Size:output:0-model_12/string_lookup_100/Equal_1/y:output:0*
T0*
_output_shapes
: Б
(model_12/string_lookup_100/Assert/AssertAssert&model_12/string_lookup_100/Equal_1:z:00model_12/string_lookup_100/StringFormat:output:0)^model_12/string_lookup_101/Assert/Assert*

T
2*&
 _has_manual_control_dependencies(*
_output_shapes
 ѕ
#model_12/string_lookup_100/IdentityIdentityAmodel_12/string_lookup_100/None_Lookup/LookupTableFindV2:values:0)^model_12/string_lookup_100/Assert/Assert*
T0	*'
_output_shapes
:€€€€€€€€€©
7model_12/string_lookup_99/None_Lookup/LookupTableFindV2LookupTableFindV2Dmodel_12_string_lookup_99_none_lookup_lookuptablefindv2_table_handlemarital_statusEmodel_12_string_lookup_99_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:€€€€€€€€€l
!model_12/string_lookup_99/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€»
model_12/string_lookup_99/EqualEqual@model_12/string_lookup_99/None_Lookup/LookupTableFindV2:values:0*model_12/string_lookup_99/Equal/y:output:0*
T0	*'
_output_shapes
:€€€€€€€€€v
model_12/string_lookup_99/WhereWhere#model_12/string_lookup_99/Equal:z:0*'
_output_shapes
:€€€€€€€€€Ђ
"model_12/string_lookup_99/GatherNdGatherNdmarital_status'model_12/string_lookup_99/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:€€€€€€€€€§
&model_12/string_lookup_99/StringFormatStringFormat+model_12/string_lookup_99/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*Е
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.p
model_12/string_lookup_99/SizeSize'model_12/string_lookup_99/Where:index:0*
T0	*
_output_shapes
: e
#model_12/string_lookup_99/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : Ґ
!model_12/string_lookup_99/Equal_1Equal'model_12/string_lookup_99/Size:output:0,model_12/string_lookup_99/Equal_1/y:output:0*
T0*
_output_shapes
: ю
'model_12/string_lookup_99/Assert/AssertAssert%model_12/string_lookup_99/Equal_1:z:0/model_12/string_lookup_99/StringFormat:output:0)^model_12/string_lookup_100/Assert/Assert*

T
2*&
 _has_manual_control_dependencies(*
_output_shapes
 ћ
"model_12/string_lookup_99/IdentityIdentity@model_12/string_lookup_99/None_Lookup/LookupTableFindV2:values:0(^model_12/string_lookup_99/Assert/Assert*
T0	*'
_output_shapes
:€€€€€€€€€±
7model_12/string_lookup_98/None_Lookup/LookupTableFindV2LookupTableFindV2Dmodel_12_string_lookup_98_none_lookup_lookuptablefindv2_table_handlecurrent_working_statusEmodel_12_string_lookup_98_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:€€€€€€€€€l
!model_12/string_lookup_98/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€»
model_12/string_lookup_98/EqualEqual@model_12/string_lookup_98/None_Lookup/LookupTableFindV2:values:0*model_12/string_lookup_98/Equal/y:output:0*
T0	*'
_output_shapes
:€€€€€€€€€v
model_12/string_lookup_98/WhereWhere#model_12/string_lookup_98/Equal:z:0*'
_output_shapes
:€€€€€€€€€≥
"model_12/string_lookup_98/GatherNdGatherNdcurrent_working_status'model_12/string_lookup_98/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:€€€€€€€€€§
&model_12/string_lookup_98/StringFormatStringFormat+model_12/string_lookup_98/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*Е
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.p
model_12/string_lookup_98/SizeSize'model_12/string_lookup_98/Where:index:0*
T0	*
_output_shapes
: e
#model_12/string_lookup_98/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : Ґ
!model_12/string_lookup_98/Equal_1Equal'model_12/string_lookup_98/Size:output:0,model_12/string_lookup_98/Equal_1/y:output:0*
T0*
_output_shapes
: э
'model_12/string_lookup_98/Assert/AssertAssert%model_12/string_lookup_98/Equal_1:z:0/model_12/string_lookup_98/StringFormat:output:0(^model_12/string_lookup_99/Assert/Assert*

T
2*&
 _has_manual_control_dependencies(*
_output_shapes
 ћ
"model_12/string_lookup_98/IdentityIdentity@model_12/string_lookup_98/None_Lookup/LookupTableFindV2:values:0(^model_12/string_lookup_98/Assert/Assert*
T0	*'
_output_shapes
:€€€€€€€€€Ґ
7model_12/string_lookup_97/None_Lookup/LookupTableFindV2LookupTableFindV2Dmodel_12_string_lookup_97_none_lookup_lookuptablefindv2_table_handleallergyEmodel_12_string_lookup_97_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:€€€€€€€€€l
!model_12/string_lookup_97/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€»
model_12/string_lookup_97/EqualEqual@model_12/string_lookup_97/None_Lookup/LookupTableFindV2:values:0*model_12/string_lookup_97/Equal/y:output:0*
T0	*'
_output_shapes
:€€€€€€€€€v
model_12/string_lookup_97/WhereWhere#model_12/string_lookup_97/Equal:z:0*'
_output_shapes
:€€€€€€€€€§
"model_12/string_lookup_97/GatherNdGatherNdallergy'model_12/string_lookup_97/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:€€€€€€€€€§
&model_12/string_lookup_97/StringFormatStringFormat+model_12/string_lookup_97/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*Е
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.p
model_12/string_lookup_97/SizeSize'model_12/string_lookup_97/Where:index:0*
T0	*
_output_shapes
: e
#model_12/string_lookup_97/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : Ґ
!model_12/string_lookup_97/Equal_1Equal'model_12/string_lookup_97/Size:output:0,model_12/string_lookup_97/Equal_1/y:output:0*
T0*
_output_shapes
: э
'model_12/string_lookup_97/Assert/AssertAssert%model_12/string_lookup_97/Equal_1:z:0/model_12/string_lookup_97/StringFormat:output:0(^model_12/string_lookup_98/Assert/Assert*

T
2*&
 _has_manual_control_dependencies(*
_output_shapes
 ћ
"model_12/string_lookup_97/IdentityIdentity@model_12/string_lookup_97/None_Lookup/LookupTableFindV2:values:0(^model_12/string_lookup_97/Assert/Assert*
T0	*'
_output_shapes
:€€€€€€€€€™
7model_12/string_lookup_96/None_Lookup/LookupTableFindV2LookupTableFindV2Dmodel_12_string_lookup_96_none_lookup_lookuptablefindv2_table_handlecultural_factorEmodel_12_string_lookup_96_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:€€€€€€€€€l
!model_12/string_lookup_96/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€»
model_12/string_lookup_96/EqualEqual@model_12/string_lookup_96/None_Lookup/LookupTableFindV2:values:0*model_12/string_lookup_96/Equal/y:output:0*
T0	*'
_output_shapes
:€€€€€€€€€v
model_12/string_lookup_96/WhereWhere#model_12/string_lookup_96/Equal:z:0*'
_output_shapes
:€€€€€€€€€ђ
"model_12/string_lookup_96/GatherNdGatherNdcultural_factor'model_12/string_lookup_96/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:€€€€€€€€€§
&model_12/string_lookup_96/StringFormatStringFormat+model_12/string_lookup_96/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*Е
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.p
model_12/string_lookup_96/SizeSize'model_12/string_lookup_96/Where:index:0*
T0	*
_output_shapes
: e
#model_12/string_lookup_96/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : Ґ
!model_12/string_lookup_96/Equal_1Equal'model_12/string_lookup_96/Size:output:0,model_12/string_lookup_96/Equal_1/y:output:0*
T0*
_output_shapes
: э
'model_12/string_lookup_96/Assert/AssertAssert%model_12/string_lookup_96/Equal_1:z:0/model_12/string_lookup_96/StringFormat:output:0(^model_12/string_lookup_97/Assert/Assert*

T
2*&
 _has_manual_control_dependencies(*
_output_shapes
 ћ
"model_12/string_lookup_96/IdentityIdentity@model_12/string_lookup_96/None_Lookup/LookupTableFindV2:values:0(^model_12/string_lookup_96/Assert/Assert*
T0	*'
_output_shapes
:€€€€€€€€€•
7model_12/string_lookup_95/None_Lookup/LookupTableFindV2LookupTableFindV2Dmodel_12_string_lookup_95_none_lookup_lookuptablefindv2_table_handle
life_styleEmodel_12_string_lookup_95_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:€€€€€€€€€l
!model_12/string_lookup_95/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€»
model_12/string_lookup_95/EqualEqual@model_12/string_lookup_95/None_Lookup/LookupTableFindV2:values:0*model_12/string_lookup_95/Equal/y:output:0*
T0	*'
_output_shapes
:€€€€€€€€€v
model_12/string_lookup_95/WhereWhere#model_12/string_lookup_95/Equal:z:0*'
_output_shapes
:€€€€€€€€€І
"model_12/string_lookup_95/GatherNdGatherNd
life_style'model_12/string_lookup_95/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:€€€€€€€€€§
&model_12/string_lookup_95/StringFormatStringFormat+model_12/string_lookup_95/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*Е
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.p
model_12/string_lookup_95/SizeSize'model_12/string_lookup_95/Where:index:0*
T0	*
_output_shapes
: e
#model_12/string_lookup_95/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : Ґ
!model_12/string_lookup_95/Equal_1Equal'model_12/string_lookup_95/Size:output:0,model_12/string_lookup_95/Equal_1/y:output:0*
T0*
_output_shapes
: э
'model_12/string_lookup_95/Assert/AssertAssert%model_12/string_lookup_95/Equal_1:z:0/model_12/string_lookup_95/StringFormat:output:0(^model_12/string_lookup_96/Assert/Assert*

T
2*&
 _has_manual_control_dependencies(*
_output_shapes
 ћ
"model_12/string_lookup_95/IdentityIdentity@model_12/string_lookup_95/None_Lookup/LookupTableFindV2:values:0(^model_12/string_lookup_95/Assert/Assert*
T0	*'
_output_shapes
:€€€€€€€€€§
7model_12/string_lookup_94/None_Lookup/LookupTableFindV2LookupTableFindV2Dmodel_12_string_lookup_94_none_lookup_lookuptablefindv2_table_handle	age_rangeEmodel_12_string_lookup_94_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:€€€€€€€€€l
!model_12/string_lookup_94/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€»
model_12/string_lookup_94/EqualEqual@model_12/string_lookup_94/None_Lookup/LookupTableFindV2:values:0*model_12/string_lookup_94/Equal/y:output:0*
T0	*'
_output_shapes
:€€€€€€€€€v
model_12/string_lookup_94/WhereWhere#model_12/string_lookup_94/Equal:z:0*'
_output_shapes
:€€€€€€€€€¶
"model_12/string_lookup_94/GatherNdGatherNd	age_range'model_12/string_lookup_94/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:€€€€€€€€€§
&model_12/string_lookup_94/StringFormatStringFormat+model_12/string_lookup_94/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*Е
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.p
model_12/string_lookup_94/SizeSize'model_12/string_lookup_94/Where:index:0*
T0	*
_output_shapes
: e
#model_12/string_lookup_94/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : Ґ
!model_12/string_lookup_94/Equal_1Equal'model_12/string_lookup_94/Size:output:0,model_12/string_lookup_94/Equal_1/y:output:0*
T0*
_output_shapes
: э
'model_12/string_lookup_94/Assert/AssertAssert%model_12/string_lookup_94/Equal_1:z:0/model_12/string_lookup_94/StringFormat:output:0(^model_12/string_lookup_95/Assert/Assert*

T
2*&
 _has_manual_control_dependencies(*
_output_shapes
 ћ
"model_12/string_lookup_94/IdentityIdentity@model_12/string_lookup_94/None_Lookup/LookupTableFindV2:values:0(^model_12/string_lookup_94/Assert/Assert*
T0	*'
_output_shapes
:€€€€€€€€€™
7model_12/string_lookup_93/None_Lookup/LookupTableFindV2LookupTableFindV2Dmodel_12_string_lookup_93_none_lookup_lookuptablefindv2_table_handleclinical_genderEmodel_12_string_lookup_93_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:€€€€€€€€€l
!model_12/string_lookup_93/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€»
model_12/string_lookup_93/EqualEqual@model_12/string_lookup_93/None_Lookup/LookupTableFindV2:values:0*model_12/string_lookup_93/Equal/y:output:0*
T0	*'
_output_shapes
:€€€€€€€€€v
model_12/string_lookup_93/WhereWhere#model_12/string_lookup_93/Equal:z:0*'
_output_shapes
:€€€€€€€€€ђ
"model_12/string_lookup_93/GatherNdGatherNdclinical_gender'model_12/string_lookup_93/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:€€€€€€€€€§
&model_12/string_lookup_93/StringFormatStringFormat+model_12/string_lookup_93/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*Е
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.p
model_12/string_lookup_93/SizeSize'model_12/string_lookup_93/Where:index:0*
T0	*
_output_shapes
: e
#model_12/string_lookup_93/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : Ґ
!model_12/string_lookup_93/Equal_1Equal'model_12/string_lookup_93/Size:output:0,model_12/string_lookup_93/Equal_1/y:output:0*
T0*
_output_shapes
: э
'model_12/string_lookup_93/Assert/AssertAssert%model_12/string_lookup_93/Equal_1:z:0/model_12/string_lookup_93/StringFormat:output:0(^model_12/string_lookup_94/Assert/Assert*

T
2*&
 _has_manual_control_dependencies(*
_output_shapes
 ћ
"model_12/string_lookup_93/IdentityIdentity@model_12/string_lookup_93/None_Lookup/LookupTableFindV2:values:0(^model_12/string_lookup_93/Assert/Assert*
T0	*'
_output_shapes
:€€€€€€€€€©
7model_12/string_lookup_92/None_Lookup/LookupTableFindV2LookupTableFindV2Dmodel_12_string_lookup_92_none_lookup_lookuptablefindv2_table_handlenutrition_goalEmodel_12_string_lookup_92_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:€€€€€€€€€l
!model_12/string_lookup_92/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€»
model_12/string_lookup_92/EqualEqual@model_12/string_lookup_92/None_Lookup/LookupTableFindV2:values:0*model_12/string_lookup_92/Equal/y:output:0*
T0	*'
_output_shapes
:€€€€€€€€€v
model_12/string_lookup_92/WhereWhere#model_12/string_lookup_92/Equal:z:0*'
_output_shapes
:€€€€€€€€€Ђ
"model_12/string_lookup_92/GatherNdGatherNdnutrition_goal'model_12/string_lookup_92/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:€€€€€€€€€§
&model_12/string_lookup_92/StringFormatStringFormat+model_12/string_lookup_92/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*Е
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.p
model_12/string_lookup_92/SizeSize'model_12/string_lookup_92/Where:index:0*
T0	*
_output_shapes
: e
#model_12/string_lookup_92/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : Ґ
!model_12/string_lookup_92/Equal_1Equal'model_12/string_lookup_92/Size:output:0,model_12/string_lookup_92/Equal_1/y:output:0*
T0*
_output_shapes
: э
'model_12/string_lookup_92/Assert/AssertAssert%model_12/string_lookup_92/Equal_1:z:0/model_12/string_lookup_92/StringFormat:output:0(^model_12/string_lookup_93/Assert/Assert*

T
2*&
 _has_manual_control_dependencies(*
_output_shapes
 ћ
"model_12/string_lookup_92/IdentityIdentity@model_12/string_lookup_92/None_Lookup/LookupTableFindV2:values:0(^model_12/string_lookup_92/Assert/Assert*
T0	*'
_output_shapes
:€€€€€€€€€¬
8model_12/string_lookup_105/None_Lookup/LookupTableFindV2LookupTableFindV2Emodel_12_string_lookup_105_none_lookup_lookuptablefindv2_table_handle$social_situation_of_meal_consumptionFmodel_12_string_lookup_105_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:€€€€€€€€€m
"model_12/string_lookup_105/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€Ћ
 model_12/string_lookup_105/EqualEqualAmodel_12/string_lookup_105/None_Lookup/LookupTableFindV2:values:0+model_12/string_lookup_105/Equal/y:output:0*
T0	*'
_output_shapes
:€€€€€€€€€x
 model_12/string_lookup_105/WhereWhere$model_12/string_lookup_105/Equal:z:0*'
_output_shapes
:€€€€€€€€€√
#model_12/string_lookup_105/GatherNdGatherNd$social_situation_of_meal_consumption(model_12/string_lookup_105/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:€€€€€€€€€¶
'model_12/string_lookup_105/StringFormatStringFormat,model_12/string_lookup_105/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*Е
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.r
model_12/string_lookup_105/SizeSize(model_12/string_lookup_105/Where:index:0*
T0	*
_output_shapes
: f
$model_12/string_lookup_105/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : •
"model_12/string_lookup_105/Equal_1Equal(model_12/string_lookup_105/Size:output:0-model_12/string_lookup_105/Equal_1/y:output:0*
T0*
_output_shapes
: А
(model_12/string_lookup_105/Assert/AssertAssert&model_12/string_lookup_105/Equal_1:z:00model_12/string_lookup_105/StringFormat:output:0(^model_12/string_lookup_92/Assert/Assert*

T
2*&
 _has_manual_control_dependencies(*
_output_shapes
 ѕ
#model_12/string_lookup_105/IdentityIdentityAmodel_12/string_lookup_105/None_Lookup/LookupTableFindV2:values:0)^model_12/string_lookup_105/Assert/Assert*
T0	*'
_output_shapes
:€€€€€€€€€Ј
8model_12/string_lookup_104/None_Lookup/LookupTableFindV2LookupTableFindV2Emodel_12_string_lookup_104_none_lookup_lookuptablefindv2_table_handleplace_of_meal_consumptionFmodel_12_string_lookup_104_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:€€€€€€€€€m
"model_12/string_lookup_104/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€Ћ
 model_12/string_lookup_104/EqualEqualAmodel_12/string_lookup_104/None_Lookup/LookupTableFindV2:values:0+model_12/string_lookup_104/Equal/y:output:0*
T0	*'
_output_shapes
:€€€€€€€€€x
 model_12/string_lookup_104/WhereWhere$model_12/string_lookup_104/Equal:z:0*'
_output_shapes
:€€€€€€€€€Є
#model_12/string_lookup_104/GatherNdGatherNdplace_of_meal_consumption(model_12/string_lookup_104/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:€€€€€€€€€¶
'model_12/string_lookup_104/StringFormatStringFormat,model_12/string_lookup_104/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*Е
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.r
model_12/string_lookup_104/SizeSize(model_12/string_lookup_104/Where:index:0*
T0	*
_output_shapes
: f
$model_12/string_lookup_104/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : •
"model_12/string_lookup_104/Equal_1Equal(model_12/string_lookup_104/Size:output:0-model_12/string_lookup_104/Equal_1/y:output:0*
T0*
_output_shapes
: Б
(model_12/string_lookup_104/Assert/AssertAssert&model_12/string_lookup_104/Equal_1:z:00model_12/string_lookup_104/StringFormat:output:0)^model_12/string_lookup_105/Assert/Assert*

T
2*&
 _has_manual_control_dependencies(*
_output_shapes
 ѕ
#model_12/string_lookup_104/IdentityIdentityAmodel_12/string_lookup_104/None_Lookup/LookupTableFindV2:values:0)^model_12/string_lookup_104/Assert/Assert*
T0	*'
_output_shapes
:€€€€€€€€€£
&model_12/embedding_90/embedding_lookupResourceGather.model_12_embedding_90_embedding_lookup_7094570,model_12/string_lookup_107/Identity:output:0*
Tindices0	*A
_class7
53loc:@model_12/embedding_90/embedding_lookup/7094570*+
_output_shapes
:€€€€€€€€€	*
dtype0Ґ
/model_12/embedding_90/embedding_lookup/IdentityIdentity/model_12/embedding_90/embedding_lookup:output:0*
T0*+
_output_shapes
:€€€€€€€€€	£
&model_12/embedding_89/embedding_lookupResourceGather.model_12_embedding_89_embedding_lookup_7094574,model_12/string_lookup_106/Identity:output:0*
Tindices0	*A
_class7
53loc:@model_12/embedding_89/embedding_lookup/7094574*+
_output_shapes
:€€€€€€€€€*
dtype0Ґ
/model_12/embedding_89/embedding_lookup/IdentityIdentity/model_12/embedding_89/embedding_lookup:output:0*
T0*+
_output_shapes
:€€€€€€€€€£
&model_12/embedding_86/embedding_lookupResourceGather.model_12_embedding_86_embedding_lookup_7094578,model_12/string_lookup_102/Identity:output:0*
Tindices0	*A
_class7
53loc:@model_12/embedding_86/embedding_lookup/7094578*+
_output_shapes
:€€€€€€€€€*
dtype0Ґ
/model_12/embedding_86/embedding_lookup/IdentityIdentity/model_12/embedding_86/embedding_lookup:output:0*
T0*+
_output_shapes
:€€€€€€€€€£
&model_12/embedding_85/embedding_lookupResourceGather.model_12_embedding_85_embedding_lookup_7094582,model_12/string_lookup_101/Identity:output:0*
Tindices0	*A
_class7
53loc:@model_12/embedding_85/embedding_lookup/7094582*+
_output_shapes
:€€€€€€€€€*
dtype0Ґ
/model_12/embedding_85/embedding_lookup/IdentityIdentity/model_12/embedding_85/embedding_lookup:output:0*
T0*+
_output_shapes
:€€€€€€€€€£
&model_12/embedding_84/embedding_lookupResourceGather.model_12_embedding_84_embedding_lookup_7094586,model_12/string_lookup_100/Identity:output:0*
Tindices0	*A
_class7
53loc:@model_12/embedding_84/embedding_lookup/7094586*+
_output_shapes
:€€€€€€€€€*
dtype0Ґ
/model_12/embedding_84/embedding_lookup/IdentityIdentity/model_12/embedding_84/embedding_lookup:output:0*
T0*+
_output_shapes
:€€€€€€€€€Ґ
&model_12/embedding_83/embedding_lookupResourceGather.model_12_embedding_83_embedding_lookup_7094590+model_12/string_lookup_99/Identity:output:0*
Tindices0	*A
_class7
53loc:@model_12/embedding_83/embedding_lookup/7094590*+
_output_shapes
:€€€€€€€€€*
dtype0Ґ
/model_12/embedding_83/embedding_lookup/IdentityIdentity/model_12/embedding_83/embedding_lookup:output:0*
T0*+
_output_shapes
:€€€€€€€€€Ґ
&model_12/embedding_82/embedding_lookupResourceGather.model_12_embedding_82_embedding_lookup_7094594+model_12/string_lookup_98/Identity:output:0*
Tindices0	*A
_class7
53loc:@model_12/embedding_82/embedding_lookup/7094594*+
_output_shapes
:€€€€€€€€€*
dtype0Ґ
/model_12/embedding_82/embedding_lookup/IdentityIdentity/model_12/embedding_82/embedding_lookup:output:0*
T0*+
_output_shapes
:€€€€€€€€€Ґ
&model_12/embedding_81/embedding_lookupResourceGather.model_12_embedding_81_embedding_lookup_7094598+model_12/string_lookup_97/Identity:output:0*
Tindices0	*A
_class7
53loc:@model_12/embedding_81/embedding_lookup/7094598*+
_output_shapes
:€€€€€€€€€*
dtype0Ґ
/model_12/embedding_81/embedding_lookup/IdentityIdentity/model_12/embedding_81/embedding_lookup:output:0*
T0*+
_output_shapes
:€€€€€€€€€Ґ
&model_12/embedding_80/embedding_lookupResourceGather.model_12_embedding_80_embedding_lookup_7094602+model_12/string_lookup_96/Identity:output:0*
Tindices0	*A
_class7
53loc:@model_12/embedding_80/embedding_lookup/7094602*+
_output_shapes
:€€€€€€€€€*
dtype0Ґ
/model_12/embedding_80/embedding_lookup/IdentityIdentity/model_12/embedding_80/embedding_lookup:output:0*
T0*+
_output_shapes
:€€€€€€€€€Ґ
&model_12/embedding_79/embedding_lookupResourceGather.model_12_embedding_79_embedding_lookup_7094606+model_12/string_lookup_95/Identity:output:0*
Tindices0	*A
_class7
53loc:@model_12/embedding_79/embedding_lookup/7094606*+
_output_shapes
:€€€€€€€€€*
dtype0Ґ
/model_12/embedding_79/embedding_lookup/IdentityIdentity/model_12/embedding_79/embedding_lookup:output:0*
T0*+
_output_shapes
:€€€€€€€€€Ґ
&model_12/embedding_78/embedding_lookupResourceGather.model_12_embedding_78_embedding_lookup_7094610+model_12/string_lookup_94/Identity:output:0*
Tindices0	*A
_class7
53loc:@model_12/embedding_78/embedding_lookup/7094610*+
_output_shapes
:€€€€€€€€€*
dtype0Ґ
/model_12/embedding_78/embedding_lookup/IdentityIdentity/model_12/embedding_78/embedding_lookup:output:0*
T0*+
_output_shapes
:€€€€€€€€€Ґ
&model_12/embedding_77/embedding_lookupResourceGather.model_12_embedding_77_embedding_lookup_7094614+model_12/string_lookup_93/Identity:output:0*
Tindices0	*A
_class7
53loc:@model_12/embedding_77/embedding_lookup/7094614*+
_output_shapes
:€€€€€€€€€*
dtype0Ґ
/model_12/embedding_77/embedding_lookup/IdentityIdentity/model_12/embedding_77/embedding_lookup:output:0*
T0*+
_output_shapes
:€€€€€€€€€Ґ
&model_12/embedding_76/embedding_lookupResourceGather.model_12_embedding_76_embedding_lookup_7094618+model_12/string_lookup_92/Identity:output:0*
Tindices0	*A
_class7
53loc:@model_12/embedding_76/embedding_lookup/7094618*+
_output_shapes
:€€€€€€€€€*
dtype0Ґ
/model_12/embedding_76/embedding_lookup/IdentityIdentity/model_12/embedding_76/embedding_lookup:output:0*
T0*+
_output_shapes
:€€€€€€€€€£
&model_12/embedding_88/embedding_lookupResourceGather.model_12_embedding_88_embedding_lookup_7094622,model_12/string_lookup_105/Identity:output:0*
Tindices0	*A
_class7
53loc:@model_12/embedding_88/embedding_lookup/7094622*+
_output_shapes
:€€€€€€€€€*
dtype0Ґ
/model_12/embedding_88/embedding_lookup/IdentityIdentity/model_12/embedding_88/embedding_lookup:output:0*
T0*+
_output_shapes
:€€€€€€€€€£
&model_12/embedding_87/embedding_lookupResourceGather.model_12_embedding_87_embedding_lookup_7094626,model_12/string_lookup_104/Identity:output:0*
Tindices0	*A
_class7
53loc:@model_12/embedding_87/embedding_lookup/7094626*+
_output_shapes
:€€€€€€€€€*
dtype0Ґ
/model_12/embedding_87/embedding_lookup/IdentityIdentity/model_12/embedding_87/embedding_lookup:output:0*
T0*+
_output_shapes
:€€€€€€€€€j
model_12/flatten_89/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ґ
model_12/flatten_89/ReshapeReshape8model_12/embedding_89/embedding_lookup/Identity:output:0"model_12/flatten_89/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€j
model_12/flatten_90/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€	   ґ
model_12/flatten_90/ReshapeReshape8model_12/embedding_90/embedding_lookup/Identity:output:0"model_12/flatten_90/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€	£
8model_12/string_lookup_108/None_Lookup/LookupTableFindV2LookupTableFindV2Emodel_12_string_lookup_108_none_lookup_lookuptablefindv2_table_handletasteFmodel_12_string_lookup_108_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:€€€€€€€€€m
"model_12/string_lookup_108/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€Ћ
 model_12/string_lookup_108/EqualEqualAmodel_12/string_lookup_108/None_Lookup/LookupTableFindV2:values:0+model_12/string_lookup_108/Equal/y:output:0*
T0	*'
_output_shapes
:€€€€€€€€€x
 model_12/string_lookup_108/WhereWhere$model_12/string_lookup_108/Equal:z:0*'
_output_shapes
:€€€€€€€€€§
#model_12/string_lookup_108/GatherNdGatherNdtaste(model_12/string_lookup_108/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:€€€€€€€€€¶
'model_12/string_lookup_108/StringFormatStringFormat,model_12/string_lookup_108/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*Е
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.r
model_12/string_lookup_108/SizeSize(model_12/string_lookup_108/Where:index:0*
T0	*
_output_shapes
: f
$model_12/string_lookup_108/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : •
"model_12/string_lookup_108/Equal_1Equal(model_12/string_lookup_108/Size:output:0-model_12/string_lookup_108/Equal_1/y:output:0*
T0*
_output_shapes
: Б
(model_12/string_lookup_108/Assert/AssertAssert&model_12/string_lookup_108/Equal_1:z:00model_12/string_lookup_108/StringFormat:output:0)^model_12/string_lookup_104/Assert/Assert*

T
2*&
 _has_manual_control_dependencies(*
_output_shapes
 ѕ
#model_12/string_lookup_108/IdentityIdentityAmodel_12/string_lookup_108/None_Lookup/LookupTableFindV2:values:0)^model_12/string_lookup_108/Assert/Assert*
T0	*'
_output_shapes
:€€€€€€€€€
(model_12/string_lookup_108/bincount/SizeSize,model_12/string_lookup_108/Identity:output:0*
T0	*
_output_shapes
: o
-model_12/string_lookup_108/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ¬
+model_12/string_lookup_108/bincount/GreaterGreater1model_12/string_lookup_108/bincount/Size:output:06model_12/string_lookup_108/bincount/Greater/y:output:0*
T0*
_output_shapes
: С
(model_12/string_lookup_108/bincount/CastCast/model_12/string_lookup_108/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: z
)model_12/string_lookup_108/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ±
'model_12/string_lookup_108/bincount/MaxMax,model_12/string_lookup_108/Identity:output:02model_12/string_lookup_108/bincount/Const:output:0*
T0	*
_output_shapes
: k
)model_12/string_lookup_108/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 RЈ
'model_12/string_lookup_108/bincount/addAddV20model_12/string_lookup_108/bincount/Max:output:02model_12/string_lookup_108/bincount/add/y:output:0*
T0	*
_output_shapes
: ™
'model_12/string_lookup_108/bincount/mulMul,model_12/string_lookup_108/bincount/Cast:y:0+model_12/string_lookup_108/bincount/add:z:0*
T0	*
_output_shapes
: o
-model_12/string_lookup_108/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 RЉ
+model_12/string_lookup_108/bincount/MaximumMaximum6model_12/string_lookup_108/bincount/minlength:output:0+model_12/string_lookup_108/bincount/mul:z:0*
T0	*
_output_shapes
: o
-model_12/string_lookup_108/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
value	B	 Rј
+model_12/string_lookup_108/bincount/MinimumMinimum6model_12/string_lookup_108/bincount/maxlength:output:0/model_12/string_lookup_108/bincount/Maximum:z:0*
T0	*
_output_shapes
: n
+model_12/string_lookup_108/bincount/Const_1Const*
_output_shapes
: *
dtype0*
valueB ™
1model_12/string_lookup_108/bincount/DenseBincountDenseBincount,model_12/string_lookup_108/Identity:output:0/model_12/string_lookup_108/bincount/Minimum:z:04model_12/string_lookup_108/bincount/Const_1:output:0*
T0*

Tidx0	*'
_output_shapes
:€€€€€€€€€*
binary_output(j
model_12/flatten_76/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ґ
model_12/flatten_76/ReshapeReshape8model_12/embedding_76/embedding_lookup/Identity:output:0"model_12/flatten_76/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€j
model_12/flatten_77/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ґ
model_12/flatten_77/ReshapeReshape8model_12/embedding_77/embedding_lookup/Identity:output:0"model_12/flatten_77/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€j
model_12/flatten_78/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ґ
model_12/flatten_78/ReshapeReshape8model_12/embedding_78/embedding_lookup/Identity:output:0"model_12/flatten_78/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€j
model_12/flatten_79/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ґ
model_12/flatten_79/ReshapeReshape8model_12/embedding_79/embedding_lookup/Identity:output:0"model_12/flatten_79/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€j
model_12/flatten_80/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ґ
model_12/flatten_80/ReshapeReshape8model_12/embedding_80/embedding_lookup/Identity:output:0"model_12/flatten_80/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€j
model_12/flatten_81/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ґ
model_12/flatten_81/ReshapeReshape8model_12/embedding_81/embedding_lookup/Identity:output:0"model_12/flatten_81/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€j
model_12/flatten_82/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ґ
model_12/flatten_82/ReshapeReshape8model_12/embedding_82/embedding_lookup/Identity:output:0"model_12/flatten_82/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€j
model_12/flatten_83/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ґ
model_12/flatten_83/ReshapeReshape8model_12/embedding_83/embedding_lookup/Identity:output:0"model_12/flatten_83/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€j
model_12/flatten_84/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ґ
model_12/flatten_84/ReshapeReshape8model_12/embedding_84/embedding_lookup/Identity:output:0"model_12/flatten_84/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€j
model_12/flatten_85/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ґ
model_12/flatten_85/ReshapeReshape8model_12/embedding_85/embedding_lookup/Identity:output:0"model_12/flatten_85/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€j
model_12/flatten_86/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ґ
model_12/flatten_86/ReshapeReshape8model_12/embedding_86/embedding_lookup/Identity:output:0"model_12/flatten_86/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€©
8model_12/string_lookup_103/None_Lookup/LookupTableFindV2LookupTableFindV2Emodel_12_string_lookup_103_none_lookup_lookuptablefindv2_table_handlemeal_type_yFmodel_12_string_lookup_103_none_lookup_lookuptablefindv2_default_value*	
Tin0*

Tout0	*'
_output_shapes
:€€€€€€€€€m
"model_12/string_lookup_103/Equal/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
€€€€€€€€€Ћ
 model_12/string_lookup_103/EqualEqualAmodel_12/string_lookup_103/None_Lookup/LookupTableFindV2:values:0+model_12/string_lookup_103/Equal/y:output:0*
T0	*'
_output_shapes
:€€€€€€€€€x
 model_12/string_lookup_103/WhereWhere$model_12/string_lookup_103/Equal:z:0*'
_output_shapes
:€€€€€€€€€™
#model_12/string_lookup_103/GatherNdGatherNdmeal_type_y(model_12/string_lookup_103/Where:index:0*
Tindices0	*
Tparams0*#
_output_shapes
:€€€€€€€€€¶
'model_12/string_lookup_103/StringFormatStringFormat,model_12/string_lookup_103/GatherNd:output:0*

T
2*
_output_shapes
: *
placeholder{}*Е
templateywWhen `num_oov_indices=0` all inputs should be in vocabulary, found OOV values {}, consider setting `num_oov_indices=1`.r
model_12/string_lookup_103/SizeSize(model_12/string_lookup_103/Where:index:0*
T0	*
_output_shapes
: f
$model_12/string_lookup_103/Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : •
"model_12/string_lookup_103/Equal_1Equal(model_12/string_lookup_103/Size:output:0-model_12/string_lookup_103/Equal_1/y:output:0*
T0*
_output_shapes
: Б
(model_12/string_lookup_103/Assert/AssertAssert&model_12/string_lookup_103/Equal_1:z:00model_12/string_lookup_103/StringFormat:output:0)^model_12/string_lookup_108/Assert/Assert*

T
2*&
 _has_manual_control_dependencies(*
_output_shapes
 ѕ
#model_12/string_lookup_103/IdentityIdentityAmodel_12/string_lookup_103/None_Lookup/LookupTableFindV2:values:0)^model_12/string_lookup_103/Assert/Assert*
T0	*'
_output_shapes
:€€€€€€€€€
(model_12/string_lookup_103/bincount/SizeSize,model_12/string_lookup_103/Identity:output:0*
T0	*
_output_shapes
: o
-model_12/string_lookup_103/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : ¬
+model_12/string_lookup_103/bincount/GreaterGreater1model_12/string_lookup_103/bincount/Size:output:06model_12/string_lookup_103/bincount/Greater/y:output:0*
T0*
_output_shapes
: С
(model_12/string_lookup_103/bincount/CastCast/model_12/string_lookup_103/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: z
)model_12/string_lookup_103/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ±
'model_12/string_lookup_103/bincount/MaxMax,model_12/string_lookup_103/Identity:output:02model_12/string_lookup_103/bincount/Const:output:0*
T0	*
_output_shapes
: k
)model_12/string_lookup_103/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 RЈ
'model_12/string_lookup_103/bincount/addAddV20model_12/string_lookup_103/bincount/Max:output:02model_12/string_lookup_103/bincount/add/y:output:0*
T0	*
_output_shapes
: ™
'model_12/string_lookup_103/bincount/mulMul,model_12/string_lookup_103/bincount/Cast:y:0+model_12/string_lookup_103/bincount/add:z:0*
T0	*
_output_shapes
: p
-model_12/string_lookup_103/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value
B	 RУЉ
+model_12/string_lookup_103/bincount/MaximumMaximum6model_12/string_lookup_103/bincount/minlength:output:0+model_12/string_lookup_103/bincount/mul:z:0*
T0	*
_output_shapes
: p
-model_12/string_lookup_103/bincount/maxlengthConst*
_output_shapes
: *
dtype0	*
value
B	 RУј
+model_12/string_lookup_103/bincount/MinimumMinimum6model_12/string_lookup_103/bincount/maxlength:output:0/model_12/string_lookup_103/bincount/Maximum:z:0*
T0	*
_output_shapes
: n
+model_12/string_lookup_103/bincount/Const_1Const*
_output_shapes
: *
dtype0*
valueB Ђ
1model_12/string_lookup_103/bincount/DenseBincountDenseBincount,model_12/string_lookup_103/Identity:output:0/model_12/string_lookup_103/bincount/Minimum:z:04model_12/string_lookup_103/bincount/Const_1:output:0*
T0*

Tidx0	*(
_output_shapes
:€€€€€€€€€У*
binary_output(j
model_12/flatten_87/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ґ
model_12/flatten_87/ReshapeReshape8model_12/embedding_87/embedding_lookup/Identity:output:0"model_12/flatten_87/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€j
model_12/flatten_88/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   ґ
model_12/flatten_88/ReshapeReshape8model_12/embedding_88/embedding_lookup/Identity:output:0"model_12/flatten_88/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€e
#model_12/concatenate_33/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ё
model_12/concatenate_33/concatConcatV2$model_12/flatten_89/Reshape:output:0calories$model_12/flatten_90/Reshape:output:0:model_12/string_lookup_108/bincount/DenseBincount:output:0pricefiberfatproteincarbohydrates
embeddings,model_12/concatenate_33/concat/axis:output:0*
N
*
T0*(
_output_shapes
:€€€€€€€€€£e
#model_12/concatenate_31/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ч
model_12/concatenate_31/concatConcatV2$model_12/flatten_76/Reshape:output:0$model_12/flatten_77/Reshape:output:0$model_12/flatten_78/Reshape:output:0$model_12/flatten_79/Reshape:output:0weightheightprojected_daily_caloriescurrent_daily_calories$model_12/flatten_80/Reshape:output:0$model_12/flatten_81/Reshape:output:0$model_12/flatten_82/Reshape:output:0$model_12/flatten_83/Reshape:output:0$model_12/flatten_84/Reshape:output:0$model_12/flatten_85/Reshape:output:0$model_12/flatten_86/Reshape:output:0,model_12/concatenate_31/concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€•
-model_12/user_embedding/MatMul/ReadVariableOpReadVariableOp6model_12_user_embedding_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0ї
model_12/user_embedding/MatMulMatMul'model_12/concatenate_31/concat:output:05model_12/user_embedding/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђ£
.model_12/user_embedding/BiasAdd/ReadVariableOpReadVariableOp7model_12_user_embedding_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0њ
model_12/user_embedding/BiasAddBiasAdd(model_12/user_embedding/MatMul:product:06model_12/user_embedding/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђ¶
-model_12/food_embedding/MatMul/ReadVariableOpReadVariableOp6model_12_food_embedding_matmul_readvariableop_resource* 
_output_shapes
:
£ђ*
dtype0ї
model_12/food_embedding/MatMulMatMul'model_12/concatenate_33/concat:output:05model_12/food_embedding/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђ£
.model_12/food_embedding/BiasAdd/ReadVariableOpReadVariableOp7model_12_food_embedding_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0њ
model_12/food_embedding/BiasAddBiasAdd(model_12/food_embedding/MatMul:product:06model_12/food_embedding/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђe
#model_12/concatenate_32/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :¬
model_12/concatenate_32/concatConcatV2
day_number:model_12/string_lookup_103/bincount/DenseBincount:output:0time_of_meal_consumption$model_12/flatten_87/Reshape:output:0$model_12/flatten_88/Reshape:output:0,model_12/concatenate_32/concat/axis:output:0*
N*
T0*(
_output_shapes
:€€€€€€€€€ШЂ
0model_12/context_embedding/MatMul/ReadVariableOpReadVariableOp9model_12_context_embedding_matmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype0ј
!model_12/context_embedding/MatMulMatMul'model_12/concatenate_32/concat:output:08model_12/context_embedding/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€®
1model_12/context_embedding/BiasAdd/ReadVariableOpReadVariableOp:model_12_context_embedding_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0«
"model_12/context_embedding/BiasAddBiasAdd+model_12/context_embedding/MatMul:product:09model_12/context_embedding/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€К
#model_12/dot_12/l2_normalize/SquareSquare(model_12/user_embedding/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђt
2model_12/dot_12/l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :–
 model_12/dot_12/l2_normalize/SumSum'model_12/dot_12/l2_normalize/Square:y:0;model_12/dot_12/l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
	keep_dims(k
&model_12/dot_12/l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *ћЉМ+љ
$model_12/dot_12/l2_normalize/MaximumMaximum)model_12/dot_12/l2_normalize/Sum:output:0/model_12/dot_12/l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€З
"model_12/dot_12/l2_normalize/RsqrtRsqrt(model_12/dot_12/l2_normalize/Maximum:z:0*
T0*'
_output_shapes
:€€€€€€€€€®
model_12/dot_12/l2_normalizeMul(model_12/user_embedding/BiasAdd:output:0&model_12/dot_12/l2_normalize/Rsqrt:y:0*
T0*(
_output_shapes
:€€€€€€€€€ђМ
%model_12/dot_12/l2_normalize_1/SquareSquare(model_12/food_embedding/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€ђv
4model_12/dot_12/l2_normalize_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :÷
"model_12/dot_12/l2_normalize_1/SumSum)model_12/dot_12/l2_normalize_1/Square:y:0=model_12/dot_12/l2_normalize_1/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
	keep_dims(m
(model_12/dot_12/l2_normalize_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *ћЉМ+√
&model_12/dot_12/l2_normalize_1/MaximumMaximum+model_12/dot_12/l2_normalize_1/Sum:output:01model_12/dot_12/l2_normalize_1/Maximum/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€Л
$model_12/dot_12/l2_normalize_1/RsqrtRsqrt*model_12/dot_12/l2_normalize_1/Maximum:z:0*
T0*'
_output_shapes
:€€€€€€€€€ђ
model_12/dot_12/l2_normalize_1Mul(model_12/food_embedding/BiasAdd:output:0(model_12/dot_12/l2_normalize_1/Rsqrt:y:0*
T0*(
_output_shapes
:€€€€€€€€€ђ`
model_12/dot_12/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :™
model_12/dot_12/ExpandDims
ExpandDims model_12/dot_12/l2_normalize:z:0'model_12/dot_12/ExpandDims/dim:output:0*
T0*,
_output_shapes
:€€€€€€€€€ђb
 model_12/dot_12/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :∞
model_12/dot_12/ExpandDims_1
ExpandDims"model_12/dot_12/l2_normalize_1:z:0)model_12/dot_12/ExpandDims_1/dim:output:0*
T0*,
_output_shapes
:€€€€€€€€€ђ©
model_12/dot_12/MatMulBatchMatMulV2#model_12/dot_12/ExpandDims:output:0%model_12/dot_12/ExpandDims_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€r
model_12/dot_12/ShapeShapemodel_12/dot_12/MatMul:output:0*
T0*
_output_shapes
::нѕМ
model_12/dot_12/SqueezeSqueezemodel_12/dot_12/MatMul:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
squeeze_dims
e
#model_12/concatenate_34/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ј
model_12/concatenate_34/concatConcatV2(model_12/user_embedding/BiasAdd:output:0(model_12/food_embedding/BiasAdd:output:0+model_12/context_embedding/BiasAdd:output:0 model_12/dot_12/Squeeze:output:0,model_12/concatenate_34/concat/axis:output:0*
N*
T0*(
_output_shapes
:€€€€€€€€€йЈ
8model_12/batch_normalization_12/batchnorm/ReadVariableOpReadVariableOpAmodel_12_batch_normalization_12_batchnorm_readvariableop_resource*
_output_shapes	
:й*
dtype0t
/model_12/batch_normalization_12/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:Ў
-model_12/batch_normalization_12/batchnorm/addAddV2@model_12/batch_normalization_12/batchnorm/ReadVariableOp:value:08model_12/batch_normalization_12/batchnorm/add/y:output:0*
T0*
_output_shapes	
:йС
/model_12/batch_normalization_12/batchnorm/RsqrtRsqrt1model_12/batch_normalization_12/batchnorm/add:z:0*
T0*
_output_shapes	
:йњ
<model_12/batch_normalization_12/batchnorm/mul/ReadVariableOpReadVariableOpEmodel_12_batch_normalization_12_batchnorm_mul_readvariableop_resource*
_output_shapes	
:й*
dtype0’
-model_12/batch_normalization_12/batchnorm/mulMul3model_12/batch_normalization_12/batchnorm/Rsqrt:y:0Dmodel_12/batch_normalization_12/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:й≈
/model_12/batch_normalization_12/batchnorm/mul_1Mul'model_12/concatenate_34/concat:output:01model_12/batch_normalization_12/batchnorm/mul:z:0*
T0*(
_output_shapes
:€€€€€€€€€йї
:model_12/batch_normalization_12/batchnorm/ReadVariableOp_1ReadVariableOpCmodel_12_batch_normalization_12_batchnorm_readvariableop_1_resource*
_output_shapes	
:й*
dtype0”
/model_12/batch_normalization_12/batchnorm/mul_2MulBmodel_12/batch_normalization_12/batchnorm/ReadVariableOp_1:value:01model_12/batch_normalization_12/batchnorm/mul:z:0*
T0*
_output_shapes	
:йї
:model_12/batch_normalization_12/batchnorm/ReadVariableOp_2ReadVariableOpCmodel_12_batch_normalization_12_batchnorm_readvariableop_2_resource*
_output_shapes	
:й*
dtype0”
-model_12/batch_normalization_12/batchnorm/subSubBmodel_12/batch_normalization_12/batchnorm/ReadVariableOp_2:value:03model_12/batch_normalization_12/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:й”
/model_12/batch_normalization_12/batchnorm/add_1AddV23model_12/batch_normalization_12/batchnorm/mul_1:z:01model_12/batch_normalization_12/batchnorm/sub:z:0*
T0*(
_output_shapes
:€€€€€€€€€йЮ
)model_12/fc_layer_0/MatMul/ReadVariableOpReadVariableOp2model_12_fc_layer_0_matmul_readvariableop_resource* 
_output_shapes
:
йА*
dtype0њ
model_12/fc_layer_0/MatMulMatMul3model_12/batch_normalization_12/batchnorm/add_1:z:01model_12/fc_layer_0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АЫ
*model_12/fc_layer_0/BiasAdd/ReadVariableOpReadVariableOp3model_12_fc_layer_0_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0≥
model_12/fc_layer_0/BiasAddBiasAdd$model_12/fc_layer_0/MatMul:product:02model_12/fc_layer_0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аy
model_12/fc_layer_0/ReluRelu$model_12/fc_layer_0/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€АЭ
)model_12/fc_layer_1/MatMul/ReadVariableOpReadVariableOp2model_12_fc_layer_1_matmul_readvariableop_resource*
_output_shapes
:	А@*
dtype0±
model_12/fc_layer_1/MatMulMatMul&model_12/fc_layer_0/Relu:activations:01model_12/fc_layer_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@Ъ
*model_12/fc_layer_1/BiasAdd/ReadVariableOpReadVariableOp3model_12_fc_layer_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0≤
model_12/fc_layer_1/BiasAddBiasAdd$model_12/fc_layer_1/MatMul:product:02model_12/fc_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@x
model_12/fc_layer_1/ReluRelu$model_12/fc_layer_1/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ь
)model_12/fc_layer_2/MatMul/ReadVariableOpReadVariableOp2model_12_fc_layer_2_matmul_readvariableop_resource*
_output_shapes

:@ *
dtype0±
model_12/fc_layer_2/MatMulMatMul&model_12/fc_layer_1/Relu:activations:01model_12/fc_layer_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ Ъ
*model_12/fc_layer_2/BiasAdd/ReadVariableOpReadVariableOp3model_12_fc_layer_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0≤
model_12/fc_layer_2/BiasAddBiasAdd$model_12/fc_layer_2/MatMul:product:02model_12/fc_layer_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ x
model_12/fc_layer_2/ReluRelu$model_12/fc_layer_2/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ Ш
'model_12/output_0/MatMul/ReadVariableOpReadVariableOp0model_12_output_0_matmul_readvariableop_resource*
_output_shapes

: *
dtype0≠
model_12/output_0/MatMulMatMul&model_12/fc_layer_2/Relu:activations:0/model_12/output_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ц
(model_12/output_0/BiasAdd/ReadVariableOpReadVariableOp1model_12_output_0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ђ
model_12/output_0/BiasAddBiasAdd"model_12/output_0/MatMul:product:00model_12/output_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€z
model_12/output_0/SigmoidSigmoid"model_12/output_0/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€l
IdentityIdentitymodel_12/output_0/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€Љ
NoOpNoOp9^model_12/batch_normalization_12/batchnorm/ReadVariableOp;^model_12/batch_normalization_12/batchnorm/ReadVariableOp_1;^model_12/batch_normalization_12/batchnorm/ReadVariableOp_2=^model_12/batch_normalization_12/batchnorm/mul/ReadVariableOp2^model_12/context_embedding/BiasAdd/ReadVariableOp1^model_12/context_embedding/MatMul/ReadVariableOp'^model_12/embedding_76/embedding_lookup'^model_12/embedding_77/embedding_lookup'^model_12/embedding_78/embedding_lookup'^model_12/embedding_79/embedding_lookup'^model_12/embedding_80/embedding_lookup'^model_12/embedding_81/embedding_lookup'^model_12/embedding_82/embedding_lookup'^model_12/embedding_83/embedding_lookup'^model_12/embedding_84/embedding_lookup'^model_12/embedding_85/embedding_lookup'^model_12/embedding_86/embedding_lookup'^model_12/embedding_87/embedding_lookup'^model_12/embedding_88/embedding_lookup'^model_12/embedding_89/embedding_lookup'^model_12/embedding_90/embedding_lookup+^model_12/fc_layer_0/BiasAdd/ReadVariableOp*^model_12/fc_layer_0/MatMul/ReadVariableOp+^model_12/fc_layer_1/BiasAdd/ReadVariableOp*^model_12/fc_layer_1/MatMul/ReadVariableOp+^model_12/fc_layer_2/BiasAdd/ReadVariableOp*^model_12/fc_layer_2/MatMul/ReadVariableOp/^model_12/food_embedding/BiasAdd/ReadVariableOp.^model_12/food_embedding/MatMul/ReadVariableOp)^model_12/output_0/BiasAdd/ReadVariableOp(^model_12/output_0/MatMul/ReadVariableOp)^model_12/string_lookup_100/Assert/Assert9^model_12/string_lookup_100/None_Lookup/LookupTableFindV2)^model_12/string_lookup_101/Assert/Assert9^model_12/string_lookup_101/None_Lookup/LookupTableFindV2)^model_12/string_lookup_102/Assert/Assert9^model_12/string_lookup_102/None_Lookup/LookupTableFindV2)^model_12/string_lookup_103/Assert/Assert9^model_12/string_lookup_103/None_Lookup/LookupTableFindV2)^model_12/string_lookup_104/Assert/Assert9^model_12/string_lookup_104/None_Lookup/LookupTableFindV2)^model_12/string_lookup_105/Assert/Assert9^model_12/string_lookup_105/None_Lookup/LookupTableFindV2)^model_12/string_lookup_106/Assert/Assert9^model_12/string_lookup_106/None_Lookup/LookupTableFindV2)^model_12/string_lookup_107/Assert/Assert9^model_12/string_lookup_107/None_Lookup/LookupTableFindV2)^model_12/string_lookup_108/Assert/Assert9^model_12/string_lookup_108/None_Lookup/LookupTableFindV2(^model_12/string_lookup_92/Assert/Assert8^model_12/string_lookup_92/None_Lookup/LookupTableFindV2(^model_12/string_lookup_93/Assert/Assert8^model_12/string_lookup_93/None_Lookup/LookupTableFindV2(^model_12/string_lookup_94/Assert/Assert8^model_12/string_lookup_94/None_Lookup/LookupTableFindV2(^model_12/string_lookup_95/Assert/Assert8^model_12/string_lookup_95/None_Lookup/LookupTableFindV2(^model_12/string_lookup_96/Assert/Assert8^model_12/string_lookup_96/None_Lookup/LookupTableFindV2(^model_12/string_lookup_97/Assert/Assert8^model_12/string_lookup_97/None_Lookup/LookupTableFindV2(^model_12/string_lookup_98/Assert/Assert8^model_12/string_lookup_98/None_Lookup/LookupTableFindV2(^model_12/string_lookup_99/Assert/Assert8^model_12/string_lookup_99/None_Lookup/LookupTableFindV2/^model_12/user_embedding/BiasAdd/ReadVariableOp.^model_12/user_embedding/MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*÷
_input_shapesƒ
Ѕ:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€А:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2t
8model_12/batch_normalization_12/batchnorm/ReadVariableOp8model_12/batch_normalization_12/batchnorm/ReadVariableOp2x
:model_12/batch_normalization_12/batchnorm/ReadVariableOp_1:model_12/batch_normalization_12/batchnorm/ReadVariableOp_12x
:model_12/batch_normalization_12/batchnorm/ReadVariableOp_2:model_12/batch_normalization_12/batchnorm/ReadVariableOp_22|
<model_12/batch_normalization_12/batchnorm/mul/ReadVariableOp<model_12/batch_normalization_12/batchnorm/mul/ReadVariableOp2f
1model_12/context_embedding/BiasAdd/ReadVariableOp1model_12/context_embedding/BiasAdd/ReadVariableOp2d
0model_12/context_embedding/MatMul/ReadVariableOp0model_12/context_embedding/MatMul/ReadVariableOp2P
&model_12/embedding_76/embedding_lookup&model_12/embedding_76/embedding_lookup2P
&model_12/embedding_77/embedding_lookup&model_12/embedding_77/embedding_lookup2P
&model_12/embedding_78/embedding_lookup&model_12/embedding_78/embedding_lookup2P
&model_12/embedding_79/embedding_lookup&model_12/embedding_79/embedding_lookup2P
&model_12/embedding_80/embedding_lookup&model_12/embedding_80/embedding_lookup2P
&model_12/embedding_81/embedding_lookup&model_12/embedding_81/embedding_lookup2P
&model_12/embedding_82/embedding_lookup&model_12/embedding_82/embedding_lookup2P
&model_12/embedding_83/embedding_lookup&model_12/embedding_83/embedding_lookup2P
&model_12/embedding_84/embedding_lookup&model_12/embedding_84/embedding_lookup2P
&model_12/embedding_85/embedding_lookup&model_12/embedding_85/embedding_lookup2P
&model_12/embedding_86/embedding_lookup&model_12/embedding_86/embedding_lookup2P
&model_12/embedding_87/embedding_lookup&model_12/embedding_87/embedding_lookup2P
&model_12/embedding_88/embedding_lookup&model_12/embedding_88/embedding_lookup2P
&model_12/embedding_89/embedding_lookup&model_12/embedding_89/embedding_lookup2P
&model_12/embedding_90/embedding_lookup&model_12/embedding_90/embedding_lookup2X
*model_12/fc_layer_0/BiasAdd/ReadVariableOp*model_12/fc_layer_0/BiasAdd/ReadVariableOp2V
)model_12/fc_layer_0/MatMul/ReadVariableOp)model_12/fc_layer_0/MatMul/ReadVariableOp2X
*model_12/fc_layer_1/BiasAdd/ReadVariableOp*model_12/fc_layer_1/BiasAdd/ReadVariableOp2V
)model_12/fc_layer_1/MatMul/ReadVariableOp)model_12/fc_layer_1/MatMul/ReadVariableOp2X
*model_12/fc_layer_2/BiasAdd/ReadVariableOp*model_12/fc_layer_2/BiasAdd/ReadVariableOp2V
)model_12/fc_layer_2/MatMul/ReadVariableOp)model_12/fc_layer_2/MatMul/ReadVariableOp2`
.model_12/food_embedding/BiasAdd/ReadVariableOp.model_12/food_embedding/BiasAdd/ReadVariableOp2^
-model_12/food_embedding/MatMul/ReadVariableOp-model_12/food_embedding/MatMul/ReadVariableOp2T
(model_12/output_0/BiasAdd/ReadVariableOp(model_12/output_0/BiasAdd/ReadVariableOp2R
'model_12/output_0/MatMul/ReadVariableOp'model_12/output_0/MatMul/ReadVariableOp2T
(model_12/string_lookup_100/Assert/Assert(model_12/string_lookup_100/Assert/Assert2t
8model_12/string_lookup_100/None_Lookup/LookupTableFindV28model_12/string_lookup_100/None_Lookup/LookupTableFindV22T
(model_12/string_lookup_101/Assert/Assert(model_12/string_lookup_101/Assert/Assert2t
8model_12/string_lookup_101/None_Lookup/LookupTableFindV28model_12/string_lookup_101/None_Lookup/LookupTableFindV22T
(model_12/string_lookup_102/Assert/Assert(model_12/string_lookup_102/Assert/Assert2t
8model_12/string_lookup_102/None_Lookup/LookupTableFindV28model_12/string_lookup_102/None_Lookup/LookupTableFindV22T
(model_12/string_lookup_103/Assert/Assert(model_12/string_lookup_103/Assert/Assert2t
8model_12/string_lookup_103/None_Lookup/LookupTableFindV28model_12/string_lookup_103/None_Lookup/LookupTableFindV22T
(model_12/string_lookup_104/Assert/Assert(model_12/string_lookup_104/Assert/Assert2t
8model_12/string_lookup_104/None_Lookup/LookupTableFindV28model_12/string_lookup_104/None_Lookup/LookupTableFindV22T
(model_12/string_lookup_105/Assert/Assert(model_12/string_lookup_105/Assert/Assert2t
8model_12/string_lookup_105/None_Lookup/LookupTableFindV28model_12/string_lookup_105/None_Lookup/LookupTableFindV22T
(model_12/string_lookup_106/Assert/Assert(model_12/string_lookup_106/Assert/Assert2t
8model_12/string_lookup_106/None_Lookup/LookupTableFindV28model_12/string_lookup_106/None_Lookup/LookupTableFindV22T
(model_12/string_lookup_107/Assert/Assert(model_12/string_lookup_107/Assert/Assert2t
8model_12/string_lookup_107/None_Lookup/LookupTableFindV28model_12/string_lookup_107/None_Lookup/LookupTableFindV22T
(model_12/string_lookup_108/Assert/Assert(model_12/string_lookup_108/Assert/Assert2t
8model_12/string_lookup_108/None_Lookup/LookupTableFindV28model_12/string_lookup_108/None_Lookup/LookupTableFindV22R
'model_12/string_lookup_92/Assert/Assert'model_12/string_lookup_92/Assert/Assert2r
7model_12/string_lookup_92/None_Lookup/LookupTableFindV27model_12/string_lookup_92/None_Lookup/LookupTableFindV22R
'model_12/string_lookup_93/Assert/Assert'model_12/string_lookup_93/Assert/Assert2r
7model_12/string_lookup_93/None_Lookup/LookupTableFindV27model_12/string_lookup_93/None_Lookup/LookupTableFindV22R
'model_12/string_lookup_94/Assert/Assert'model_12/string_lookup_94/Assert/Assert2r
7model_12/string_lookup_94/None_Lookup/LookupTableFindV27model_12/string_lookup_94/None_Lookup/LookupTableFindV22R
'model_12/string_lookup_95/Assert/Assert'model_12/string_lookup_95/Assert/Assert2r
7model_12/string_lookup_95/None_Lookup/LookupTableFindV27model_12/string_lookup_95/None_Lookup/LookupTableFindV22R
'model_12/string_lookup_96/Assert/Assert'model_12/string_lookup_96/Assert/Assert2r
7model_12/string_lookup_96/None_Lookup/LookupTableFindV27model_12/string_lookup_96/None_Lookup/LookupTableFindV22R
'model_12/string_lookup_97/Assert/Assert'model_12/string_lookup_97/Assert/Assert2r
7model_12/string_lookup_97/None_Lookup/LookupTableFindV27model_12/string_lookup_97/None_Lookup/LookupTableFindV22R
'model_12/string_lookup_98/Assert/Assert'model_12/string_lookup_98/Assert/Assert2r
7model_12/string_lookup_98/None_Lookup/LookupTableFindV27model_12/string_lookup_98/None_Lookup/LookupTableFindV22R
'model_12/string_lookup_99/Assert/Assert'model_12/string_lookup_99/Assert/Assert2r
7model_12/string_lookup_99/None_Lookup/LookupTableFindV27model_12/string_lookup_99/None_Lookup/LookupTableFindV22`
.model_12/user_embedding/BiasAdd/ReadVariableOp.model_12/user_embedding/BiasAdd/ReadVariableOp2^
-model_12/user_embedding/MatMul/ReadVariableOp-model_12/user_embedding/MatMul/ReadVariableOp:L H
'
_output_shapes
:€€€€€€€€€

_user_specified_nameBMI:RN
'
_output_shapes
:€€€€€€€€€
#
_user_specified_name	age_range:RN
'
_output_shapes
:€€€€€€€€€
#
_user_specified_name	allergens:PL
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	allergy:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
calories:VR
'
_output_shapes
:€€€€€€€€€
'
_user_specified_namecarbohydrates:XT
'
_output_shapes
:€€€€€€€€€
)
_user_specified_nameclinical_gender:XT
'
_output_shapes
:€€€€€€€€€
)
_user_specified_namecultural_factor:]Y
'
_output_shapes
:€€€€€€€€€
.
_user_specified_namecultural_restriction:_	[
'
_output_shapes
:€€€€€€€€€
0
_user_specified_namecurrent_daily_calories:_
[
'
_output_shapes
:€€€€€€€€€
0
_user_specified_namecurrent_working_status:SO
'
_output_shapes
:€€€€€€€€€
$
_user_specified_name
day_number:TP
(
_output_shapes
:€€€€€€€€€А
$
_user_specified_name
embeddings:RN
'
_output_shapes
:€€€€€€€€€
#
_user_specified_name	ethnicity:LH
'
_output_shapes
:€€€€€€€€€

_user_specified_namefat:NJ
'
_output_shapes
:€€€€€€€€€

_user_specified_namefiber:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameheight:SO
'
_output_shapes
:€€€€€€€€€
$
_user_specified_name
life_style:WS
'
_output_shapes
:€€€€€€€€€
(
_user_specified_namemarital_status:TP
'
_output_shapes
:€€€€€€€€€
%
_user_specified_namemeal_type_y:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
next_BMI:WS
'
_output_shapes
:€€€€€€€€€
(
_user_specified_namenutrition_goal:b^
'
_output_shapes
:€€€€€€€€€
3
_user_specified_nameplace_of_meal_consumption:NJ
'
_output_shapes
:€€€€€€€€€

_user_specified_nameprice:a]
'
_output_shapes
:€€€€€€€€€
2
_user_specified_nameprojected_daily_calories:PL
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	protein:mi
'
_output_shapes
:€€€€€€€€€
>
_user_specified_name&$social_situation_of_meal_consumption:NJ
'
_output_shapes
:€€€€€€€€€

_user_specified_nametaste:a]
'
_output_shapes
:€€€€€€€€€
2
_user_specified_nametime_of_meal_consumption:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameweight:,(
&
_user_specified_nametable_handle:

_output_shapes
: :, (
&
_user_specified_nametable_handle:!

_output_shapes
: :,"(
&
_user_specified_nametable_handle:#

_output_shapes
: :,$(
&
_user_specified_nametable_handle:%

_output_shapes
: :,&(
&
_user_specified_nametable_handle:'

_output_shapes
: :,((
&
_user_specified_nametable_handle:)

_output_shapes
: :,*(
&
_user_specified_nametable_handle:+

_output_shapes
: :,,(
&
_user_specified_nametable_handle:-

_output_shapes
: :,.(
&
_user_specified_nametable_handle:/

_output_shapes
: :,0(
&
_user_specified_nametable_handle:1

_output_shapes
: :,2(
&
_user_specified_nametable_handle:3

_output_shapes
: :,4(
&
_user_specified_nametable_handle:5

_output_shapes
: :,6(
&
_user_specified_nametable_handle:7

_output_shapes
: :,8(
&
_user_specified_nametable_handle:9

_output_shapes
: :,:(
&
_user_specified_nametable_handle:;

_output_shapes
: :'<#
!
_user_specified_name	7094570:'=#
!
_user_specified_name	7094574:'>#
!
_user_specified_name	7094578:'?#
!
_user_specified_name	7094582:'@#
!
_user_specified_name	7094586:'A#
!
_user_specified_name	7094590:'B#
!
_user_specified_name	7094594:'C#
!
_user_specified_name	7094598:'D#
!
_user_specified_name	7094602:'E#
!
_user_specified_name	7094606:'F#
!
_user_specified_name	7094610:'G#
!
_user_specified_name	7094614:'H#
!
_user_specified_name	7094618:'I#
!
_user_specified_name	7094622:'J#
!
_user_specified_name	7094626:,K(
&
_user_specified_nametable_handle:L

_output_shapes
: :,M(
&
_user_specified_nametable_handle:N

_output_shapes
: :(O$
"
_user_specified_name
resource:(P$
"
_user_specified_name
resource:(Q$
"
_user_specified_name
resource:(R$
"
_user_specified_name
resource:(S$
"
_user_specified_name
resource:(T$
"
_user_specified_name
resource:(U$
"
_user_specified_name
resource:(V$
"
_user_specified_name
resource:(W$
"
_user_specified_name
resource:(X$
"
_user_specified_name
resource:(Y$
"
_user_specified_name
resource:(Z$
"
_user_specified_name
resource:([$
"
_user_specified_name
resource:(\$
"
_user_specified_name
resource:(]$
"
_user_specified_name
resource:(^$
"
_user_specified_name
resource:(_$
"
_user_specified_name
resource:(`$
"
_user_specified_name
resource
∞
M
$__inference__update_step_xla_7097073
gradient
variable:	ђ*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
	:ђ: *
	_noinline(:E A

_output_shapes	
:ђ
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Ь
.
__inference__destroyer_7098125
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
ѓ
<
__inference__creator_7098054
identityИҐ
hash_tableo

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	6434660*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: /
NoOpNoOp^hash_table*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
°
І
I__inference_embedding_89_layer_call_and_return_conditional_losses_7095116

inputs	*
embedding_lookup_7095111:
identityИҐembedding_lookupї
embedding_lookupResourceGatherembedding_lookup_7095111inputs*
Tindices0	*+
_class!
loc:@embedding_lookup/7095111*+
_output_shapes
:€€€€€€€€€*
dtype0v
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_output_shapes
:€€€€€€€€€u
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€5
NoOpNoOp^embedding_lookup*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:€€€€€€€€€: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:'#
!
_user_specified_name	7095111
°
І
I__inference_embedding_84_layer_call_and_return_conditional_losses_7097268

inputs	*
embedding_lookup_7097263:
identityИҐembedding_lookupї
embedding_lookupResourceGatherembedding_lookup_7097263inputs*
Tindices0	*+
_class!
loc:@embedding_lookup/7097263*+
_output_shapes
:€€€€€€€€€*
dtype0v
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_output_shapes
:€€€€€€€€€u
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€5
NoOpNoOp^embedding_lookup*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:€€€€€€€€€: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:'#
!
_user_specified_name	7097263
Ь
.
__inference__destroyer_7098215
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
ƒ
°
K__inference_concatenate_32_layer_call_and_return_conditional_losses_7097656
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ц
concatConcatV2inputs_0inputs_1inputs_2inputs_3inputs_4concat/axis:output:0*
N*
T0*(
_output_shapes
:€€€€€€€€€ШX
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:€€€€€€€€€Ш"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*s
_input_shapesb
`:€€€€€€€€€:€€€€€€€€€У:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:Q M
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_0:RN
(
_output_shapes
:€€€€€€€€€У
"
_user_specified_name
inputs_1:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_2:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_3:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_4
щ
Щ
,__inference_fc_layer_2_layer_call_fn_7097865

inputs
unknown:@ 
	unknown_0: 
identityИҐStatefulPartitionedCallя
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_fc_layer_2_layer_call_and_return_conditional_losses_7095627o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs:'#
!
_user_specified_name	7097859:'#
!
_user_specified_name	7097861
ѓ
<
__inference__creator_7098114
identityИҐ
hash_tableo

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	6434864*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: /
NoOpNoOp^hash_table*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
°
І
I__inference_embedding_88_layer_call_and_return_conditional_losses_7097501

inputs	*
embedding_lookup_7097496:
identityИҐembedding_lookupї
embedding_lookupResourceGatherembedding_lookup_7097496inputs*
Tindices0	*+
_class!
loc:@embedding_lookup/7097496*+
_output_shapes
:€€€€€€€€€*
dtype0v
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_output_shapes
:€€€€€€€€€u
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€5
NoOpNoOp^embedding_lookup*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:€€€€€€€€€: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:'#
!
_user_specified_name	7097496
Ь
.
__inference__destroyer_7098020
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
Љ
Q
$__inference__update_step_xla_7097078
gradient
variable:	Ш*
_XlaMustCompile(*(
_construction_contextkEagerRuntime* 
_input_shapes
:	Ш: *
	_noinline(:I E

_output_shapes
:	Ш
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
ѓ
Е
 __inference__initializer_7098136:
6key_value_init6435138_lookuptableimportv2_table_handle2
.key_value_init6435138_lookuptableimportv2_keys4
0key_value_init6435138_lookuptableimportv2_values	
identityИҐ)key_value_init6435138/LookupTableImportV2З
)key_value_init6435138/LookupTableImportV2LookupTableImportV26key_value_init6435138_lookuptableimportv2_table_handle.key_value_init6435138_lookuptableimportv2_keys0key_value_init6435138_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: N
NoOpNoOp*^key_value_init6435138/LookupTableImportV2*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2V
)key_value_init6435138/LookupTableImportV2)key_value_init6435138/LookupTableImportV2:, (
&
_user_specified_nametable_handle: 

_output_shapes
:: 

_output_shapes
:
Х
p
$__inference__update_step_xla_7096990
gradient

gradient_1	

gradient_2
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:€€€€€€€€€:€€€€€€€€€:: *
	_noinline(:Q M
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
gradient:MI
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
gradient:D@

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
м
±
G__inference_fc_layer_0_layer_call_and_return_conditional_losses_7097832

inputs2
matmul_readvariableop_resource:
йА.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐ3fc_layer_0/kernel/Regularizer/L2Loss/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
йА*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€АФ
3fc_layer_0/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
йА*
dtype0М
$fc_layer_0/kernel/Regularizer/L2LossL2Loss;fc_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#fc_layer_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„£<¶
!fc_layer_0/kernel/Regularizer/mulMul,fc_layer_0/kernel/Regularizer/mul/x:output:0-fc_layer_0/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€АЙ
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp4^fc_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€й: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2j
3fc_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp3fc_layer_0/kernel/Regularizer/L2Loss/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€й
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Ь
.
__inference__destroyer_7098050
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
°
І
I__inference_embedding_77_layer_call_and_return_conditional_losses_7097163

inputs	*
embedding_lookup_7097158:
identityИҐembedding_lookupї
embedding_lookupResourceGatherembedding_lookup_7097158inputs*
Tindices0	*+
_class!
loc:@embedding_lookup/7097158*+
_output_shapes
:€€€€€€€€€*
dtype0v
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_output_shapes
:€€€€€€€€€u
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€5
NoOpNoOp^embedding_lookup*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:€€€€€€€€€: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:'#
!
_user_specified_name	7097158
ѓ
Е
 __inference__initializer_7098031:
6key_value_init6434557_lookuptableimportv2_table_handle2
.key_value_init6434557_lookuptableimportv2_keys4
0key_value_init6434557_lookuptableimportv2_values	
identityИҐ)key_value_init6434557/LookupTableImportV2З
)key_value_init6434557/LookupTableImportV2LookupTableImportV26key_value_init6434557_lookuptableimportv2_table_handle.key_value_init6434557_lookuptableimportv2_keys0key_value_init6434557_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: N
NoOpNoOp*^key_value_init6434557/LookupTableImportV2*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2V
)key_value_init6434557/LookupTableImportV2)key_value_init6434557/LookupTableImportV2:, (
&
_user_specified_nametable_handle: 

_output_shapes
:: 

_output_shapes
:
Ь
.
__inference__destroyer_7098140
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
њ
c
G__inference_flatten_80_layer_call_and_return_conditional_losses_7095337

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:€€€€€€€€€X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
°
І
I__inference_embedding_83_layer_call_and_return_conditional_losses_7097253

inputs	*
embedding_lookup_7097248:
identityИҐembedding_lookupї
embedding_lookupResourceGatherembedding_lookup_7097248inputs*
Tindices0	*+
_class!
loc:@embedding_lookup/7097248*+
_output_shapes
:€€€€€€€€€*
dtype0v
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_output_shapes
:€€€€€€€€€u
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€5
NoOpNoOp^embedding_lookup*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:€€€€€€€€€: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:'#
!
_user_specified_name	7097248
і
В
.__inference_embedding_82_layer_call_fn_7097230

inputs	
unknown:
identityИҐStatefulPartitionedCallЎ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_embedding_82_layer_call_and_return_conditional_losses_7095171s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:€€€€€€€€€: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:'#
!
_user_specified_name	7097226
і
В
.__inference_embedding_80_layer_call_fn_7097200

inputs	
unknown:
identityИҐStatefulPartitionedCallЎ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_embedding_80_layer_call_and_return_conditional_losses_7095193s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:€€€€€€€€€: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:'#
!
_user_specified_name	7097196
ѓ
Е
 __inference__initializer_7098196:
6key_value_init6435240_lookuptableimportv2_table_handle2
.key_value_init6435240_lookuptableimportv2_keys4
0key_value_init6435240_lookuptableimportv2_values	
identityИҐ)key_value_init6435240/LookupTableImportV2З
)key_value_init6435240/LookupTableImportV2LookupTableImportV26key_value_init6435240_lookuptableimportv2_table_handle.key_value_init6435240_lookuptableimportv2_keys0key_value_init6435240_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: N
NoOpNoOp*^key_value_init6435240/LookupTableImportV2*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2V
)key_value_init6435240/LookupTableImportV2)key_value_init6435240/LookupTableImportV2:, (
&
_user_specified_nametable_handle: 

_output_shapes
:: 

_output_shapes
:
°
І
I__inference_embedding_89_layer_call_and_return_conditional_losses_7097313

inputs	*
embedding_lookup_7097308:
identityИҐembedding_lookupї
embedding_lookupResourceGatherembedding_lookup_7097308inputs*
Tindices0	*+
_class!
loc:@embedding_lookup/7097308*+
_output_shapes
:€€€€€€€€€*
dtype0v
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_output_shapes
:€€€€€€€€€u
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€5
NoOpNoOp^embedding_lookup*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:€€€€€€€€€: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:'#
!
_user_specified_name	7097308
°
І
I__inference_embedding_82_layer_call_and_return_conditional_losses_7097238

inputs	*
embedding_lookup_7097233:
identityИҐembedding_lookupї
embedding_lookupResourceGatherembedding_lookup_7097233inputs*
Tindices0	*+
_class!
loc:@embedding_lookup/7097233*+
_output_shapes
:€€€€€€€€€*
dtype0v
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_output_shapes
:€€€€€€€€€u
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€5
NoOpNoOp^embedding_lookup*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:€€€€€€€€€: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:'#
!
_user_specified_name	7097233
°
І
I__inference_embedding_79_layer_call_and_return_conditional_losses_7095204

inputs	*
embedding_lookup_7095199:
identityИҐembedding_lookupї
embedding_lookupResourceGatherembedding_lookup_7095199inputs*
Tindices0	*+
_class!
loc:@embedding_lookup/7095199*+
_output_shapes
:€€€€€€€€€*
dtype0v
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_output_shapes
:€€€€€€€€€u
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€5
NoOpNoOp^embedding_lookup*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:€€€€€€€€€: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:'#
!
_user_specified_name	7095199
ь
Ъ
,__inference_fc_layer_1_layer_call_fn_7097841

inputs
unknown:	А@
	unknown_0:@
identityИҐStatefulPartitionedCallя
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_fc_layer_1_layer_call_and_return_conditional_losses_7095607o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs:'#
!
_user_specified_name	7097835:'#
!
_user_specified_name	7097837
і
В
.__inference_embedding_84_layer_call_fn_7097260

inputs	
unknown:
identityИҐStatefulPartitionedCallЎ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_embedding_84_layer_call_and_return_conditional_losses_7095149s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:€€€€€€€€€: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:'#
!
_user_specified_name	7097256
њ
c
G__inference_flatten_90_layer_call_and_return_conditional_losses_7097471

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€	   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:€€€€€€€€€	X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€	:S O
+
_output_shapes
:€€€€€€€€€	
 
_user_specified_nameinputs
°
І
I__inference_embedding_85_layer_call_and_return_conditional_losses_7095138

inputs	*
embedding_lookup_7095133:
identityИҐembedding_lookupї
embedding_lookupResourceGatherembedding_lookup_7095133inputs*
Tindices0	*+
_class!
loc:@embedding_lookup/7095133*+
_output_shapes
:€€€€€€€€€*
dtype0v
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_output_shapes
:€€€€€€€€€u
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€5
NoOpNoOp^embedding_lookup*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:€€€€€€€€€: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:'#
!
_user_specified_name	7095133
є
P
$__inference__update_step_xla_7097128
gradient
variable: *
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
: : *
	_noinline(:H D

_output_shapes

: 
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
∞
M
$__inference__update_step_xla_7097063
gradient
variable:	ђ*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
	:ђ: *
	_noinline(:E A

_output_shapes	
:ђ
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Ь
.
__inference__destroyer_7098005
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
Ѓ
H
,__inference_flatten_90_layer_call_fn_7097465

inputs
identityµ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_flatten_90_layer_call_and_return_conditional_losses_7095275`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€	"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€	:S O
+
_output_shapes
:€€€€€€€€€	
 
_user_specified_nameinputs
≠
L
$__inference__update_step_xla_7097113
gradient
variable:@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:@: *
	_noinline(:D @

_output_shapes
:@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
і
В
.__inference_embedding_83_layer_call_fn_7097245

inputs	
unknown:
identityИҐStatefulPartitionedCallЎ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_embedding_83_layer_call_and_return_conditional_losses_7095160s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:€€€€€€€€€: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:'#
!
_user_specified_name	7097241
®N
’
*__inference_model_12_layer_call_fn_7096254
bmi
	age_range
	allergens
allergy
calories
carbohydrates
clinical_gender
cultural_factor
cultural_restriction
current_daily_calories
current_working_status

day_number

embeddings
	ethnicity
fat	
fiber

height

life_style
marital_status
meal_type_y
next_bmi
nutrition_goal
place_of_meal_consumption	
price
projected_daily_calories
protein(
$social_situation_of_meal_consumption	
taste
time_of_meal_consumption

weight
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3
	unknown_4	
	unknown_5
	unknown_6	
	unknown_7
	unknown_8	
	unknown_9

unknown_10	

unknown_11

unknown_12	

unknown_13

unknown_14	

unknown_15

unknown_16	

unknown_17

unknown_18	

unknown_19

unknown_20	

unknown_21

unknown_22	

unknown_23

unknown_24	

unknown_25

unknown_26	

unknown_27

unknown_28	

unknown_29:W	

unknown_30:

unknown_31:

unknown_32:

unknown_33:

unknown_34:

unknown_35:

unknown_36:

unknown_37:

unknown_38:

unknown_39:

unknown_40:

unknown_41:

unknown_42:

unknown_43:

unknown_44

unknown_45	

unknown_46

unknown_47	

unknown_48:	ђ

unknown_49:	ђ

unknown_50:
£ђ

unknown_51:	ђ

unknown_52:	Ш

unknown_53:

unknown_54:	й

unknown_55:	й

unknown_56:	й

unknown_57:	й

unknown_58:
йА

unknown_59:	А

unknown_60:	А@

unknown_61:@

unknown_62:@ 

unknown_63: 

unknown_64: 

unknown_65:
identityИҐStatefulPartitionedCallЖ
StatefulPartitionedCallStatefulPartitionedCallbmi	age_range	allergensallergycaloriescarbohydratesclinical_gendercultural_factorcultural_restrictioncurrent_daily_caloriescurrent_working_status
day_number
embeddings	ethnicityfatfiberheight
life_stylemarital_statusmeal_type_ynext_bminutrition_goalplace_of_meal_consumptionpriceprojected_daily_caloriesprotein$social_situation_of_meal_consumptiontastetime_of_meal_consumptionweightunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65*l
Tine
c2a																	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*A
_read_only_resource_inputs#
!<=>?@ABCDEFGHIJOPQRSTWXYZ[\]^_`*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_model_12_layer_call_and_return_conditional_losses_7095682o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*÷
_input_shapesƒ
Ѕ:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€А:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
'
_output_shapes
:€€€€€€€€€

_user_specified_nameBMI:RN
'
_output_shapes
:€€€€€€€€€
#
_user_specified_name	age_range:RN
'
_output_shapes
:€€€€€€€€€
#
_user_specified_name	allergens:PL
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	allergy:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
calories:VR
'
_output_shapes
:€€€€€€€€€
'
_user_specified_namecarbohydrates:XT
'
_output_shapes
:€€€€€€€€€
)
_user_specified_nameclinical_gender:XT
'
_output_shapes
:€€€€€€€€€
)
_user_specified_namecultural_factor:]Y
'
_output_shapes
:€€€€€€€€€
.
_user_specified_namecultural_restriction:_	[
'
_output_shapes
:€€€€€€€€€
0
_user_specified_namecurrent_daily_calories:_
[
'
_output_shapes
:€€€€€€€€€
0
_user_specified_namecurrent_working_status:SO
'
_output_shapes
:€€€€€€€€€
$
_user_specified_name
day_number:TP
(
_output_shapes
:€€€€€€€€€А
$
_user_specified_name
embeddings:RN
'
_output_shapes
:€€€€€€€€€
#
_user_specified_name	ethnicity:LH
'
_output_shapes
:€€€€€€€€€

_user_specified_namefat:NJ
'
_output_shapes
:€€€€€€€€€

_user_specified_namefiber:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameheight:SO
'
_output_shapes
:€€€€€€€€€
$
_user_specified_name
life_style:WS
'
_output_shapes
:€€€€€€€€€
(
_user_specified_namemarital_status:TP
'
_output_shapes
:€€€€€€€€€
%
_user_specified_namemeal_type_y:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
next_BMI:WS
'
_output_shapes
:€€€€€€€€€
(
_user_specified_namenutrition_goal:b^
'
_output_shapes
:€€€€€€€€€
3
_user_specified_nameplace_of_meal_consumption:NJ
'
_output_shapes
:€€€€€€€€€

_user_specified_nameprice:a]
'
_output_shapes
:€€€€€€€€€
2
_user_specified_nameprojected_daily_calories:PL
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	protein:mi
'
_output_shapes
:€€€€€€€€€
>
_user_specified_name&$social_situation_of_meal_consumption:NJ
'
_output_shapes
:€€€€€€€€€

_user_specified_nametaste:a]
'
_output_shapes
:€€€€€€€€€
2
_user_specified_nametime_of_meal_consumption:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameweight:'#
!
_user_specified_name	7096118:

_output_shapes
: :' #
!
_user_specified_name	7096122:!

_output_shapes
: :'"#
!
_user_specified_name	7096126:#

_output_shapes
: :'$#
!
_user_specified_name	7096130:%

_output_shapes
: :'&#
!
_user_specified_name	7096134:'

_output_shapes
: :'(#
!
_user_specified_name	7096138:)

_output_shapes
: :'*#
!
_user_specified_name	7096142:+

_output_shapes
: :',#
!
_user_specified_name	7096146:-

_output_shapes
: :'.#
!
_user_specified_name	7096150:/

_output_shapes
: :'0#
!
_user_specified_name	7096154:1

_output_shapes
: :'2#
!
_user_specified_name	7096158:3

_output_shapes
: :'4#
!
_user_specified_name	7096162:5

_output_shapes
: :'6#
!
_user_specified_name	7096166:7

_output_shapes
: :'8#
!
_user_specified_name	7096170:9

_output_shapes
: :':#
!
_user_specified_name	7096174:;

_output_shapes
: :'<#
!
_user_specified_name	7096178:'=#
!
_user_specified_name	7096180:'>#
!
_user_specified_name	7096182:'?#
!
_user_specified_name	7096184:'@#
!
_user_specified_name	7096186:'A#
!
_user_specified_name	7096188:'B#
!
_user_specified_name	7096190:'C#
!
_user_specified_name	7096192:'D#
!
_user_specified_name	7096194:'E#
!
_user_specified_name	7096196:'F#
!
_user_specified_name	7096198:'G#
!
_user_specified_name	7096200:'H#
!
_user_specified_name	7096202:'I#
!
_user_specified_name	7096204:'J#
!
_user_specified_name	7096206:'K#
!
_user_specified_name	7096208:L

_output_shapes
: :'M#
!
_user_specified_name	7096212:N

_output_shapes
: :'O#
!
_user_specified_name	7096216:'P#
!
_user_specified_name	7096218:'Q#
!
_user_specified_name	7096220:'R#
!
_user_specified_name	7096222:'S#
!
_user_specified_name	7096224:'T#
!
_user_specified_name	7096226:'U#
!
_user_specified_name	7096228:'V#
!
_user_specified_name	7096230:'W#
!
_user_specified_name	7096232:'X#
!
_user_specified_name	7096234:'Y#
!
_user_specified_name	7096236:'Z#
!
_user_specified_name	7096238:'[#
!
_user_specified_name	7096240:'\#
!
_user_specified_name	7096242:']#
!
_user_specified_name	7096244:'^#
!
_user_specified_name	7096246:'_#
!
_user_specified_name	7096248:'`#
!
_user_specified_name	7096250
є
Ч
0__inference_concatenate_31_layer_call_fn_7097520
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
	inputs_10
	inputs_11
	inputs_12
	inputs_13
	inputs_14
identityЏ
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11	inputs_12	inputs_13	inputs_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_concatenate_31_layer_call_and_return_conditional_losses_7095457`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*≤
_input_shapes†
Э:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:Q M
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_1:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_2:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_3:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_4:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_5:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_6:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_7:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_8:Q	M
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_9:R
N
'
_output_shapes
:€€€€€€€€€
#
_user_specified_name	inputs_10:RN
'
_output_shapes
:€€€€€€€€€
#
_user_specified_name	inputs_11:RN
'
_output_shapes
:€€€€€€€€€
#
_user_specified_name	inputs_12:RN
'
_output_shapes
:€€€€€€€€€
#
_user_specified_name	inputs_13:RN
'
_output_shapes
:€€€€€€€€€
#
_user_specified_name	inputs_14
Ѓ
H
,__inference_flatten_81_layer_call_fn_7097388

inputs
identityµ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_flatten_81_layer_call_and_return_conditional_losses_7095344`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
≠
L
$__inference__update_step_xla_7097133
gradient
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

:: *
	_noinline(:D @

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
њ
c
G__inference_flatten_86_layer_call_and_return_conditional_losses_7095379

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:€€€€€€€€€X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
њ
c
G__inference_flatten_79_layer_call_and_return_conditional_losses_7095330

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:€€€€€€€€€X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Х
p
$__inference__update_step_xla_7096997
gradient

gradient_1	

gradient_2
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:€€€€€€€€€:€€€€€€€€€:: *
	_noinline(:Q M
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
gradient:MI
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
gradient:D@

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
і
В
.__inference_embedding_85_layer_call_fn_7097275

inputs	
unknown:
identityИҐStatefulPartitionedCallЎ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_embedding_85_layer_call_and_return_conditional_losses_7095138s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:€€€€€€€€€: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:'#
!
_user_specified_name	7097271
ж
ґ
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_7097808

inputs0
!batchnorm_readvariableop_resource:	й4
%batchnorm_mul_readvariableop_resource:	й2
#batchnorm_readvariableop_1_resource:	й2
#batchnorm_readvariableop_2_resource:	й
identityИҐbatchnorm/ReadVariableOpҐbatchnorm/ReadVariableOp_1Ґbatchnorm/ReadVariableOp_2Ґbatchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:й*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:йQ
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:й
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:й*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:йd
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:€€€€€€€€€й{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:й*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:й{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:й*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:йs
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:€€€€€€€€€йc
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€йЦ
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€й: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€й
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
і
В
.__inference_embedding_87_layer_call_fn_7097478

inputs	
unknown:
identityИҐStatefulPartitionedCallЎ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_embedding_87_layer_call_and_return_conditional_losses_7095259s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:€€€€€€€€€: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:'#
!
_user_specified_name	7097474
Љ
Q
$__inference__update_step_xla_7097058
gradient
variable:	ђ*
_XlaMustCompile(*(
_construction_contextkEagerRuntime* 
_input_shapes
:	ђ: *
	_noinline(:I E

_output_shapes
:	ђ
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Ь
.
__inference__destroyer_7098185
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
Х
p
$__inference__update_step_xla_7097004
gradient

gradient_1	

gradient_2
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:€€€€€€€€€:€€€€€€€€€:: *
	_noinline(:Q M
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
gradient:MI
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
gradient:D@

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
ѓ
<
__inference__creator_7098039
identityИҐ
hash_tableo

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	6434609*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: /
NoOpNoOp^hash_table*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
Ѓ
H
,__inference_flatten_78_layer_call_fn_7097355

inputs
identityµ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_flatten_78_layer_call_and_return_conditional_losses_7095323`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ь
.
__inference__destroyer_7098170
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
Х
p
$__inference__update_step_xla_7096983
gradient

gradient_1	

gradient_2
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:€€€€€€€€€:€€€€€€€€€:: *
	_noinline(:Q M
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
gradient:MI
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
gradient:D@

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
°
І
I__inference_embedding_78_layer_call_and_return_conditional_losses_7095215

inputs	*
embedding_lookup_7095210:
identityИҐembedding_lookupї
embedding_lookupResourceGatherembedding_lookup_7095210inputs*
Tindices0	*+
_class!
loc:@embedding_lookup/7095210*+
_output_shapes
:€€€€€€€€€*
dtype0v
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_output_shapes
:€€€€€€€€€u
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€5
NoOpNoOp^embedding_lookup*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:€€€€€€€€€: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:'#
!
_user_specified_name	7095210
њ
R
$__inference__update_step_xla_7097098
gradient
variable:
йА*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*!
_input_shapes
:
йА: *
	_noinline(:J F
 
_output_shapes
:
йА
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Е
Я
0__inference_user_embedding_layer_call_fn_7097600

inputs
unknown:	ђ
	unknown_0:	ђ
identityИҐStatefulPartitionedCallд
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€ђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_user_embedding_layer_call_and_return_conditional_losses_7095472p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€ђ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:'#
!
_user_specified_name	7097594:'#
!
_user_specified_name	7097596
Х
p
$__inference__update_step_xla_7097018
gradient

gradient_1	

gradient_2
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:€€€€€€€€€:€€€€€€€€€:: *
	_noinline(:Q M
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
gradient:MI
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
gradient:D@

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
ѓ
<
__inference__creator_7098099
identityИҐ
hash_tableo

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	6434813*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: /
NoOpNoOp^hash_table*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
њ
c
G__inference_flatten_89_layer_call_and_return_conditional_losses_7095268

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:€€€€€€€€€X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ь
.
__inference__destroyer_7097990
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
њ
c
G__inference_flatten_86_layer_call_and_return_conditional_losses_7097449

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:€€€€€€€€€X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
°
І
I__inference_embedding_86_layer_call_and_return_conditional_losses_7097298

inputs	*
embedding_lookup_7097293:
identityИҐembedding_lookupї
embedding_lookupResourceGatherembedding_lookup_7097293inputs*
Tindices0	*+
_class!
loc:@embedding_lookup/7097293*+
_output_shapes
:€€€€€€€€€*
dtype0v
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_output_shapes
:€€€€€€€€€u
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€5
NoOpNoOp^embedding_lookup*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:€€€€€€€€€: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:'#
!
_user_specified_name	7097293
і
В
.__inference_embedding_90_layer_call_fn_7097320

inputs	
unknown:W	
identityИҐStatefulPartitionedCallЎ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€	*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_embedding_90_layer_call_and_return_conditional_losses_7095105s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€	<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:€€€€€€€€€: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:'#
!
_user_specified_name	7097316
І

≈
__inference_loss_fn_2_7097928V
Ccontext_embedding_kernel_regularizer_l2loss_readvariableop_resource:	Ш
identityИҐ:context_embedding/kernel/Regularizer/L2Loss/ReadVariableOpњ
:context_embedding/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpCcontext_embedding_kernel_regularizer_l2loss_readvariableop_resource*
_output_shapes
:	Ш*
dtype0Ъ
+context_embedding/kernel/Regularizer/L2LossL2LossBcontext_embedding/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: o
*context_embedding/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„£<ї
(context_embedding/kernel/Regularizer/mulMul3context_embedding/kernel/Regularizer/mul/x:output:04context_embedding/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: j
IdentityIdentity,context_embedding/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: _
NoOpNoOp;^context_embedding/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2x
:context_embedding/kernel/Regularizer/L2Loss/ReadVariableOp:context_embedding/kernel/Regularizer/L2Loss/ReadVariableOp:( $
"
_user_specified_name
resource
ѓ
<
__inference__creator_7098084
identityИҐ
hash_tableo

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	6434762*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: /
NoOpNoOp^hash_table*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
Ь
.
__inference__destroyer_7098035
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
њ
c
G__inference_flatten_77_layer_call_and_return_conditional_losses_7095316

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:€€€€€€€€€X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
х
Е
 __inference__initializer_7098211:
6key_value_init6434958_lookuptableimportv2_table_handle2
.key_value_init6434958_lookuptableimportv2_keys4
0key_value_init6434958_lookuptableimportv2_values	
identityИҐ)key_value_init6434958/LookupTableImportV2З
)key_value_init6434958/LookupTableImportV2LookupTableImportV26key_value_init6434958_lookuptableimportv2_table_handle.key_value_init6434958_lookuptableimportv2_keys0key_value_init6434958_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: N
NoOpNoOp*^key_value_init6434958/LookupTableImportV2*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
: :У:У2V
)key_value_init6434958/LookupTableImportV2)key_value_init6434958/LookupTableImportV2:, (
&
_user_specified_nametable_handle:A=

_output_shapes	
:У

_user_specified_namekeys:C?

_output_shapes	
:У
 
_user_specified_namevalues
≠
L
$__inference__update_step_xla_7097123
gradient
variable: *
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes

: : *
	_noinline(:D @

_output_shapes
: 
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
њ
c
G__inference_flatten_80_layer_call_and_return_conditional_losses_7097383

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:€€€€€€€€€X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ѓ
Е
 __inference__initializer_7097971:
6key_value_init6434353_lookuptableimportv2_table_handle2
.key_value_init6434353_lookuptableimportv2_keys4
0key_value_init6434353_lookuptableimportv2_values	
identityИҐ)key_value_init6434353/LookupTableImportV2З
)key_value_init6434353/LookupTableImportV2LookupTableImportV26key_value_init6434353_lookuptableimportv2_table_handle.key_value_init6434353_lookuptableimportv2_keys0key_value_init6434353_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: N
NoOpNoOp*^key_value_init6434353/LookupTableImportV2*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2V
)key_value_init6434353/LookupTableImportV2)key_value_init6434353/LookupTableImportV2:, (
&
_user_specified_nametable_handle: 

_output_shapes
:: 

_output_shapes
:
™N
’
*__inference_model_12_layer_call_fn_7096422
bmi
	age_range
	allergens
allergy
calories
carbohydrates
clinical_gender
cultural_factor
cultural_restriction
current_daily_calories
current_working_status

day_number

embeddings
	ethnicity
fat	
fiber

height

life_style
marital_status
meal_type_y
next_bmi
nutrition_goal
place_of_meal_consumption	
price
projected_daily_calories
protein(
$social_situation_of_meal_consumption	
taste
time_of_meal_consumption

weight
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3
	unknown_4	
	unknown_5
	unknown_6	
	unknown_7
	unknown_8	
	unknown_9

unknown_10	

unknown_11

unknown_12	

unknown_13

unknown_14	

unknown_15

unknown_16	

unknown_17

unknown_18	

unknown_19

unknown_20	

unknown_21

unknown_22	

unknown_23

unknown_24	

unknown_25

unknown_26	

unknown_27

unknown_28	

unknown_29:W	

unknown_30:

unknown_31:

unknown_32:

unknown_33:

unknown_34:

unknown_35:

unknown_36:

unknown_37:

unknown_38:

unknown_39:

unknown_40:

unknown_41:

unknown_42:

unknown_43:

unknown_44

unknown_45	

unknown_46

unknown_47	

unknown_48:	ђ

unknown_49:	ђ

unknown_50:
£ђ

unknown_51:	ђ

unknown_52:	Ш

unknown_53:

unknown_54:	й

unknown_55:	й

unknown_56:	й

unknown_57:	й

unknown_58:
йА

unknown_59:	А

unknown_60:	А@

unknown_61:@

unknown_62:@ 

unknown_63: 

unknown_64: 

unknown_65:
identityИҐStatefulPartitionedCallИ
StatefulPartitionedCallStatefulPartitionedCallbmi	age_range	allergensallergycaloriescarbohydratesclinical_gendercultural_factorcultural_restrictioncurrent_daily_caloriescurrent_working_status
day_number
embeddings	ethnicityfatfiberheight
life_stylemarital_statusmeal_type_ynext_bminutrition_goalplace_of_meal_consumptionpriceprojected_daily_caloriesprotein$social_situation_of_meal_consumptiontastetime_of_meal_consumptionweightunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62
unknown_63
unknown_64
unknown_65*l
Tine
c2a																	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*C
_read_only_resource_inputs%
#!<=>?@ABCDEFGHIJOPQRSTUVWXYZ[\]^_`*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_model_12_layer_call_and_return_conditional_losses_7096086o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*÷
_input_shapesƒ
Ѕ:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€А:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
'
_output_shapes
:€€€€€€€€€

_user_specified_nameBMI:RN
'
_output_shapes
:€€€€€€€€€
#
_user_specified_name	age_range:RN
'
_output_shapes
:€€€€€€€€€
#
_user_specified_name	allergens:PL
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	allergy:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
calories:VR
'
_output_shapes
:€€€€€€€€€
'
_user_specified_namecarbohydrates:XT
'
_output_shapes
:€€€€€€€€€
)
_user_specified_nameclinical_gender:XT
'
_output_shapes
:€€€€€€€€€
)
_user_specified_namecultural_factor:]Y
'
_output_shapes
:€€€€€€€€€
.
_user_specified_namecultural_restriction:_	[
'
_output_shapes
:€€€€€€€€€
0
_user_specified_namecurrent_daily_calories:_
[
'
_output_shapes
:€€€€€€€€€
0
_user_specified_namecurrent_working_status:SO
'
_output_shapes
:€€€€€€€€€
$
_user_specified_name
day_number:TP
(
_output_shapes
:€€€€€€€€€А
$
_user_specified_name
embeddings:RN
'
_output_shapes
:€€€€€€€€€
#
_user_specified_name	ethnicity:LH
'
_output_shapes
:€€€€€€€€€

_user_specified_namefat:NJ
'
_output_shapes
:€€€€€€€€€

_user_specified_namefiber:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameheight:SO
'
_output_shapes
:€€€€€€€€€
$
_user_specified_name
life_style:WS
'
_output_shapes
:€€€€€€€€€
(
_user_specified_namemarital_status:TP
'
_output_shapes
:€€€€€€€€€
%
_user_specified_namemeal_type_y:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
next_BMI:WS
'
_output_shapes
:€€€€€€€€€
(
_user_specified_namenutrition_goal:b^
'
_output_shapes
:€€€€€€€€€
3
_user_specified_nameplace_of_meal_consumption:NJ
'
_output_shapes
:€€€€€€€€€

_user_specified_nameprice:a]
'
_output_shapes
:€€€€€€€€€
2
_user_specified_nameprojected_daily_calories:PL
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	protein:mi
'
_output_shapes
:€€€€€€€€€
>
_user_specified_name&$social_situation_of_meal_consumption:NJ
'
_output_shapes
:€€€€€€€€€

_user_specified_nametaste:a]
'
_output_shapes
:€€€€€€€€€
2
_user_specified_nametime_of_meal_consumption:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameweight:'#
!
_user_specified_name	7096286:

_output_shapes
: :' #
!
_user_specified_name	7096290:!

_output_shapes
: :'"#
!
_user_specified_name	7096294:#

_output_shapes
: :'$#
!
_user_specified_name	7096298:%

_output_shapes
: :'&#
!
_user_specified_name	7096302:'

_output_shapes
: :'(#
!
_user_specified_name	7096306:)

_output_shapes
: :'*#
!
_user_specified_name	7096310:+

_output_shapes
: :',#
!
_user_specified_name	7096314:-

_output_shapes
: :'.#
!
_user_specified_name	7096318:/

_output_shapes
: :'0#
!
_user_specified_name	7096322:1

_output_shapes
: :'2#
!
_user_specified_name	7096326:3

_output_shapes
: :'4#
!
_user_specified_name	7096330:5

_output_shapes
: :'6#
!
_user_specified_name	7096334:7

_output_shapes
: :'8#
!
_user_specified_name	7096338:9

_output_shapes
: :':#
!
_user_specified_name	7096342:;

_output_shapes
: :'<#
!
_user_specified_name	7096346:'=#
!
_user_specified_name	7096348:'>#
!
_user_specified_name	7096350:'?#
!
_user_specified_name	7096352:'@#
!
_user_specified_name	7096354:'A#
!
_user_specified_name	7096356:'B#
!
_user_specified_name	7096358:'C#
!
_user_specified_name	7096360:'D#
!
_user_specified_name	7096362:'E#
!
_user_specified_name	7096364:'F#
!
_user_specified_name	7096366:'G#
!
_user_specified_name	7096368:'H#
!
_user_specified_name	7096370:'I#
!
_user_specified_name	7096372:'J#
!
_user_specified_name	7096374:'K#
!
_user_specified_name	7096376:L

_output_shapes
: :'M#
!
_user_specified_name	7096380:N

_output_shapes
: :'O#
!
_user_specified_name	7096384:'P#
!
_user_specified_name	7096386:'Q#
!
_user_specified_name	7096388:'R#
!
_user_specified_name	7096390:'S#
!
_user_specified_name	7096392:'T#
!
_user_specified_name	7096394:'U#
!
_user_specified_name	7096396:'V#
!
_user_specified_name	7096398:'W#
!
_user_specified_name	7096400:'X#
!
_user_specified_name	7096402:'Y#
!
_user_specified_name	7096404:'Z#
!
_user_specified_name	7096406:'[#
!
_user_specified_name	7096408:'\#
!
_user_specified_name	7096410:']#
!
_user_specified_name	7096412:'^#
!
_user_specified_name	7096414:'_#
!
_user_specified_name	7096416:'`#
!
_user_specified_name	7096418
њ
c
G__inference_flatten_82_layer_call_and_return_conditional_losses_7097405

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:€€€€€€€€€X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Х
p
$__inference__update_step_xla_7096969
gradient

gradient_1	

gradient_2
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:€€€€€€€€€:€€€€€€€€€:: *
	_noinline(:Q M
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
gradient:MI
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
gradient:D@

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
ѓ
<
__inference__creator_7098204
identityИҐ
hash_tableo

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	6434959*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: /
NoOpNoOp^hash_table*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
њ
c
G__inference_flatten_87_layer_call_and_return_conditional_losses_7097580

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:€€€€€€€€€X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
г
ѓ
G__inference_fc_layer_1_layer_call_and_return_conditional_losses_7095607

inputs1
matmul_readvariableop_resource:	А@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐ3fc_layer_1/kernel/Regularizer/L2Loss/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@У
3fc_layer_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А@*
dtype0М
$fc_layer_1/kernel/Regularizer/L2LossL2Loss;fc_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#fc_layer_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„£<¶
!fc_layer_1/kernel/Regularizer/mulMul,fc_layer_1/kernel/Regularizer/mul/x:output:0-fc_layer_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@Й
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp4^fc_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2j
3fc_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp3fc_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
э	
њ
__inference_loss_fn_0_7097912S
@user_embedding_kernel_regularizer_l2loss_readvariableop_resource:	ђ
identityИҐ7user_embedding/kernel/Regularizer/L2Loss/ReadVariableOpє
7user_embedding/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp@user_embedding_kernel_regularizer_l2loss_readvariableop_resource*
_output_shapes
:	ђ*
dtype0Ф
(user_embedding/kernel/Regularizer/L2LossL2Loss?user_embedding/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: l
'user_embedding/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„£<≤
%user_embedding/kernel/Regularizer/mulMul0user_embedding/kernel/Regularizer/mul/x:output:01user_embedding/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: g
IdentityIdentity)user_embedding/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: \
NoOpNoOp8^user_embedding/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2r
7user_embedding/kernel/Regularizer/L2Loss/ReadVariableOp7user_embedding/kernel/Regularizer/L2Loss/ReadVariableOp:( $
"
_user_specified_name
resource
њ
c
G__inference_flatten_84_layer_call_and_return_conditional_losses_7097427

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:€€€€€€€€€X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
њ
c
G__inference_flatten_76_layer_call_and_return_conditional_losses_7095309

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:€€€€€€€€€X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ѓ
H
,__inference_flatten_84_layer_call_fn_7097421

inputs
identityµ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_flatten_84_layer_call_and_return_conditional_losses_7095365`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Љ
Q
$__inference__update_step_xla_7097108
gradient
variable:	А@*
_XlaMustCompile(*(
_construction_contextkEagerRuntime* 
_input_shapes
:	А@: *
	_noinline(:I E

_output_shapes
:	А@
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Њ
≤
K__inference_concatenate_31_layer_call_and_return_conditional_losses_7097540
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
	inputs_10
	inputs_11
	inputs_12
	inputs_13
	inputs_14
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ю
concatConcatV2inputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11	inputs_12	inputs_13	inputs_14concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*≤
_input_shapes†
Э:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:Q M
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_1:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_2:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_3:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_4:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_5:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_6:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_7:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_8:Q	M
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_9:R
N
'
_output_shapes
:€€€€€€€€€
#
_user_specified_name	inputs_10:RN
'
_output_shapes
:€€€€€€€€€
#
_user_specified_name	inputs_11:RN
'
_output_shapes
:€€€€€€€€€
#
_user_specified_name	inputs_12:RN
'
_output_shapes
:€€€€€€€€€
#
_user_specified_name	inputs_13:RN
'
_output_shapes
:€€€€€€€€€
#
_user_specified_name	inputs_14
Х
p
$__inference__update_step_xla_7097032
gradient

gradient_1	

gradient_2
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:€€€€€€€€€:€€€€€€€€€:: *
	_noinline(:Q M
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
gradient:MI
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
gradient:D@

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
≈
™
E__inference_output_0_layer_call_and_return_conditional_losses_7095647

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐ1output_0/kernel/Regularizer/L2Loss/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€Р
1output_0/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0И
"output_0/kernel/Regularizer/L2LossL2Loss9output_0/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!output_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„£<†
output_0/kernel/Regularizer/mulMul*output_0/kernel/Regularizer/mul/x:output:0+output_0/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€З
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^output_0/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1output_0/kernel/Regularizer/L2Loss/ReadVariableOp1output_0/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Х
p
$__inference__update_step_xla_7097046
gradient

gradient_1	

gradient_2
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:€€€€€€€€€:€€€€€€€€€:: *
	_noinline(:Q M
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
gradient:MI
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
gradient:D@

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Ь
.
__inference__destroyer_7098080
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
Он
ЯF
#__inference__traced_restore_7099305
file_prefix:
(assignvariableop_embedding_76_embeddings:<
*assignvariableop_1_embedding_77_embeddings:<
*assignvariableop_2_embedding_78_embeddings:<
*assignvariableop_3_embedding_79_embeddings:<
*assignvariableop_4_embedding_80_embeddings:<
*assignvariableop_5_embedding_81_embeddings:<
*assignvariableop_6_embedding_82_embeddings:<
*assignvariableop_7_embedding_83_embeddings:<
*assignvariableop_8_embedding_84_embeddings:<
*assignvariableop_9_embedding_85_embeddings:=
+assignvariableop_10_embedding_86_embeddings:=
+assignvariableop_11_embedding_89_embeddings:=
+assignvariableop_12_embedding_90_embeddings:W	=
+assignvariableop_13_embedding_87_embeddings:=
+assignvariableop_14_embedding_88_embeddings:<
)assignvariableop_15_user_embedding_kernel:	ђ6
'assignvariableop_16_user_embedding_bias:	ђ=
)assignvariableop_17_food_embedding_kernel:
£ђ6
'assignvariableop_18_food_embedding_bias:	ђ?
,assignvariableop_19_context_embedding_kernel:	Ш8
*assignvariableop_20_context_embedding_bias:?
0assignvariableop_21_batch_normalization_12_gamma:	й>
/assignvariableop_22_batch_normalization_12_beta:	йE
6assignvariableop_23_batch_normalization_12_moving_mean:	йI
:assignvariableop_24_batch_normalization_12_moving_variance:	й9
%assignvariableop_25_fc_layer_0_kernel:
йА2
#assignvariableop_26_fc_layer_0_bias:	А8
%assignvariableop_27_fc_layer_1_kernel:	А@1
#assignvariableop_28_fc_layer_1_bias:@7
%assignvariableop_29_fc_layer_2_kernel:@ 1
#assignvariableop_30_fc_layer_2_bias: 5
#assignvariableop_31_output_0_kernel: /
!assignvariableop_32_output_0_bias:'
assignvariableop_33_iteration:	 +
!assignvariableop_34_learning_rate: D
2assignvariableop_35_adam_m_embedding_76_embeddings:D
2assignvariableop_36_adam_v_embedding_76_embeddings:D
2assignvariableop_37_adam_m_embedding_77_embeddings:D
2assignvariableop_38_adam_v_embedding_77_embeddings:D
2assignvariableop_39_adam_m_embedding_78_embeddings:D
2assignvariableop_40_adam_v_embedding_78_embeddings:D
2assignvariableop_41_adam_m_embedding_79_embeddings:D
2assignvariableop_42_adam_v_embedding_79_embeddings:D
2assignvariableop_43_adam_m_embedding_80_embeddings:D
2assignvariableop_44_adam_v_embedding_80_embeddings:D
2assignvariableop_45_adam_m_embedding_81_embeddings:D
2assignvariableop_46_adam_v_embedding_81_embeddings:D
2assignvariableop_47_adam_m_embedding_82_embeddings:D
2assignvariableop_48_adam_v_embedding_82_embeddings:D
2assignvariableop_49_adam_m_embedding_83_embeddings:D
2assignvariableop_50_adam_v_embedding_83_embeddings:D
2assignvariableop_51_adam_m_embedding_84_embeddings:D
2assignvariableop_52_adam_v_embedding_84_embeddings:D
2assignvariableop_53_adam_m_embedding_85_embeddings:D
2assignvariableop_54_adam_v_embedding_85_embeddings:D
2assignvariableop_55_adam_m_embedding_86_embeddings:D
2assignvariableop_56_adam_v_embedding_86_embeddings:D
2assignvariableop_57_adam_m_embedding_89_embeddings:D
2assignvariableop_58_adam_v_embedding_89_embeddings:D
2assignvariableop_59_adam_m_embedding_90_embeddings:W	D
2assignvariableop_60_adam_v_embedding_90_embeddings:W	D
2assignvariableop_61_adam_m_embedding_87_embeddings:D
2assignvariableop_62_adam_v_embedding_87_embeddings:D
2assignvariableop_63_adam_m_embedding_88_embeddings:D
2assignvariableop_64_adam_v_embedding_88_embeddings:C
0assignvariableop_65_adam_m_user_embedding_kernel:	ђC
0assignvariableop_66_adam_v_user_embedding_kernel:	ђ=
.assignvariableop_67_adam_m_user_embedding_bias:	ђ=
.assignvariableop_68_adam_v_user_embedding_bias:	ђD
0assignvariableop_69_adam_m_food_embedding_kernel:
£ђD
0assignvariableop_70_adam_v_food_embedding_kernel:
£ђ=
.assignvariableop_71_adam_m_food_embedding_bias:	ђ=
.assignvariableop_72_adam_v_food_embedding_bias:	ђF
3assignvariableop_73_adam_m_context_embedding_kernel:	ШF
3assignvariableop_74_adam_v_context_embedding_kernel:	Ш?
1assignvariableop_75_adam_m_context_embedding_bias:?
1assignvariableop_76_adam_v_context_embedding_bias:F
7assignvariableop_77_adam_m_batch_normalization_12_gamma:	йF
7assignvariableop_78_adam_v_batch_normalization_12_gamma:	йE
6assignvariableop_79_adam_m_batch_normalization_12_beta:	йE
6assignvariableop_80_adam_v_batch_normalization_12_beta:	й@
,assignvariableop_81_adam_m_fc_layer_0_kernel:
йА@
,assignvariableop_82_adam_v_fc_layer_0_kernel:
йА9
*assignvariableop_83_adam_m_fc_layer_0_bias:	А9
*assignvariableop_84_adam_v_fc_layer_0_bias:	А?
,assignvariableop_85_adam_m_fc_layer_1_kernel:	А@?
,assignvariableop_86_adam_v_fc_layer_1_kernel:	А@8
*assignvariableop_87_adam_m_fc_layer_1_bias:@8
*assignvariableop_88_adam_v_fc_layer_1_bias:@>
,assignvariableop_89_adam_m_fc_layer_2_kernel:@ >
,assignvariableop_90_adam_v_fc_layer_2_kernel:@ 8
*assignvariableop_91_adam_m_fc_layer_2_bias: 8
*assignvariableop_92_adam_v_fc_layer_2_bias: <
*assignvariableop_93_adam_m_output_0_kernel: <
*assignvariableop_94_adam_v_output_0_kernel: 6
(assignvariableop_95_adam_m_output_0_bias:6
(assignvariableop_96_adam_v_output_0_bias:%
assignvariableop_97_total_1: %
assignvariableop_98_count_1: #
assignvariableop_99_total: $
assignvariableop_100_count: 3
%assignvariableop_101_true_positives_1:2
$assignvariableop_102_false_positives:1
#assignvariableop_103_true_positives:2
$assignvariableop_104_false_negatives:
identity_106ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_100ҐAssignVariableOp_101ҐAssignVariableOp_102ҐAssignVariableOp_103ҐAssignVariableOp_104ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_24ҐAssignVariableOp_25ҐAssignVariableOp_26ҐAssignVariableOp_27ҐAssignVariableOp_28ҐAssignVariableOp_29ҐAssignVariableOp_3ҐAssignVariableOp_30ҐAssignVariableOp_31ҐAssignVariableOp_32ҐAssignVariableOp_33ҐAssignVariableOp_34ҐAssignVariableOp_35ҐAssignVariableOp_36ҐAssignVariableOp_37ҐAssignVariableOp_38ҐAssignVariableOp_39ҐAssignVariableOp_4ҐAssignVariableOp_40ҐAssignVariableOp_41ҐAssignVariableOp_42ҐAssignVariableOp_43ҐAssignVariableOp_44ҐAssignVariableOp_45ҐAssignVariableOp_46ҐAssignVariableOp_47ҐAssignVariableOp_48ҐAssignVariableOp_49ҐAssignVariableOp_5ҐAssignVariableOp_50ҐAssignVariableOp_51ҐAssignVariableOp_52ҐAssignVariableOp_53ҐAssignVariableOp_54ҐAssignVariableOp_55ҐAssignVariableOp_56ҐAssignVariableOp_57ҐAssignVariableOp_58ҐAssignVariableOp_59ҐAssignVariableOp_6ҐAssignVariableOp_60ҐAssignVariableOp_61ҐAssignVariableOp_62ҐAssignVariableOp_63ҐAssignVariableOp_64ҐAssignVariableOp_65ҐAssignVariableOp_66ҐAssignVariableOp_67ҐAssignVariableOp_68ҐAssignVariableOp_69ҐAssignVariableOp_7ҐAssignVariableOp_70ҐAssignVariableOp_71ҐAssignVariableOp_72ҐAssignVariableOp_73ҐAssignVariableOp_74ҐAssignVariableOp_75ҐAssignVariableOp_76ҐAssignVariableOp_77ҐAssignVariableOp_78ҐAssignVariableOp_79ҐAssignVariableOp_8ҐAssignVariableOp_80ҐAssignVariableOp_81ҐAssignVariableOp_82ҐAssignVariableOp_83ҐAssignVariableOp_84ҐAssignVariableOp_85ҐAssignVariableOp_86ҐAssignVariableOp_87ҐAssignVariableOp_88ҐAssignVariableOp_89ҐAssignVariableOp_9ҐAssignVariableOp_90ҐAssignVariableOp_91ҐAssignVariableOp_92ҐAssignVariableOp_93ҐAssignVariableOp_94ҐAssignVariableOp_95ҐAssignVariableOp_96ҐAssignVariableOp_97ҐAssignVariableOp_98ҐAssignVariableOp_99б-
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:j*
dtype0*З-
valueэ,Bъ,jB:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-2/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-3/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-4/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-5/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-6/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-7/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-8/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-9/embeddings/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-10/embeddings/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-11/embeddings/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-12/embeddings/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-13/embeddings/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-14/embeddings/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-18/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-18/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-18/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-21/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-22/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/49/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/50/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/51/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/52/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/53/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/54/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/55/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/56/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/57/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/58/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/59/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/60/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/61/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/62/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH«
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:j*
dtype0*й
valueяB№jB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ≥
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Њ
_output_shapesЂ
®::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*x
dtypesn
l2j	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:ї
AssignVariableOpAssignVariableOp(assignvariableop_embedding_76_embeddingsIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_1AssignVariableOp*assignvariableop_1_embedding_77_embeddingsIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_2AssignVariableOp*assignvariableop_2_embedding_78_embeddingsIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_3AssignVariableOp*assignvariableop_3_embedding_79_embeddingsIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_4AssignVariableOp*assignvariableop_4_embedding_80_embeddingsIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_5AssignVariableOp*assignvariableop_5_embedding_81_embeddingsIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_6AssignVariableOp*assignvariableop_6_embedding_82_embeddingsIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_7AssignVariableOp*assignvariableop_7_embedding_83_embeddingsIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_8AssignVariableOp*assignvariableop_8_embedding_84_embeddingsIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_9AssignVariableOp*assignvariableop_9_embedding_85_embeddingsIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:ƒ
AssignVariableOp_10AssignVariableOp+assignvariableop_10_embedding_86_embeddingsIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:ƒ
AssignVariableOp_11AssignVariableOp+assignvariableop_11_embedding_89_embeddingsIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:ƒ
AssignVariableOp_12AssignVariableOp+assignvariableop_12_embedding_90_embeddingsIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:ƒ
AssignVariableOp_13AssignVariableOp+assignvariableop_13_embedding_87_embeddingsIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:ƒ
AssignVariableOp_14AssignVariableOp+assignvariableop_14_embedding_88_embeddingsIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_15AssignVariableOp)assignvariableop_15_user_embedding_kernelIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:ј
AssignVariableOp_16AssignVariableOp'assignvariableop_16_user_embedding_biasIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_17AssignVariableOp)assignvariableop_17_food_embedding_kernelIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:ј
AssignVariableOp_18AssignVariableOp'assignvariableop_18_food_embedding_biasIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:≈
AssignVariableOp_19AssignVariableOp,assignvariableop_19_context_embedding_kernelIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:√
AssignVariableOp_20AssignVariableOp*assignvariableop_20_context_embedding_biasIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:…
AssignVariableOp_21AssignVariableOp0assignvariableop_21_batch_normalization_12_gammaIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:»
AssignVariableOp_22AssignVariableOp/assignvariableop_22_batch_normalization_12_betaIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:ѕ
AssignVariableOp_23AssignVariableOp6assignvariableop_23_batch_normalization_12_moving_meanIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:”
AssignVariableOp_24AssignVariableOp:assignvariableop_24_batch_normalization_12_moving_varianceIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:Њ
AssignVariableOp_25AssignVariableOp%assignvariableop_25_fc_layer_0_kernelIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:Љ
AssignVariableOp_26AssignVariableOp#assignvariableop_26_fc_layer_0_biasIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:Њ
AssignVariableOp_27AssignVariableOp%assignvariableop_27_fc_layer_1_kernelIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:Љ
AssignVariableOp_28AssignVariableOp#assignvariableop_28_fc_layer_1_biasIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:Њ
AssignVariableOp_29AssignVariableOp%assignvariableop_29_fc_layer_2_kernelIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:Љ
AssignVariableOp_30AssignVariableOp#assignvariableop_30_fc_layer_2_biasIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:Љ
AssignVariableOp_31AssignVariableOp#assignvariableop_31_output_0_kernelIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_32AssignVariableOp!assignvariableop_32_output_0_biasIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0	*
_output_shapes
:ґ
AssignVariableOp_33AssignVariableOpassignvariableop_33_iterationIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_34AssignVariableOp!assignvariableop_34_learning_rateIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_35AssignVariableOp2assignvariableop_35_adam_m_embedding_76_embeddingsIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_36AssignVariableOp2assignvariableop_36_adam_v_embedding_76_embeddingsIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_37AssignVariableOp2assignvariableop_37_adam_m_embedding_77_embeddingsIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_38AssignVariableOp2assignvariableop_38_adam_v_embedding_77_embeddingsIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_39AssignVariableOp2assignvariableop_39_adam_m_embedding_78_embeddingsIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_40AssignVariableOp2assignvariableop_40_adam_v_embedding_78_embeddingsIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_41AssignVariableOp2assignvariableop_41_adam_m_embedding_79_embeddingsIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_42AssignVariableOp2assignvariableop_42_adam_v_embedding_79_embeddingsIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_43AssignVariableOp2assignvariableop_43_adam_m_embedding_80_embeddingsIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_44AssignVariableOp2assignvariableop_44_adam_v_embedding_80_embeddingsIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_45AssignVariableOp2assignvariableop_45_adam_m_embedding_81_embeddingsIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_46AssignVariableOp2assignvariableop_46_adam_v_embedding_81_embeddingsIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_47AssignVariableOp2assignvariableop_47_adam_m_embedding_82_embeddingsIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_48AssignVariableOp2assignvariableop_48_adam_v_embedding_82_embeddingsIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_49AssignVariableOp2assignvariableop_49_adam_m_embedding_83_embeddingsIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_50AssignVariableOp2assignvariableop_50_adam_v_embedding_83_embeddingsIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_51AssignVariableOp2assignvariableop_51_adam_m_embedding_84_embeddingsIdentity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_52AssignVariableOp2assignvariableop_52_adam_v_embedding_84_embeddingsIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_53AssignVariableOp2assignvariableop_53_adam_m_embedding_85_embeddingsIdentity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_54AssignVariableOp2assignvariableop_54_adam_v_embedding_85_embeddingsIdentity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_55AssignVariableOp2assignvariableop_55_adam_m_embedding_86_embeddingsIdentity_55:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_56AssignVariableOp2assignvariableop_56_adam_v_embedding_86_embeddingsIdentity_56:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_57AssignVariableOp2assignvariableop_57_adam_m_embedding_89_embeddingsIdentity_57:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_58AssignVariableOp2assignvariableop_58_adam_v_embedding_89_embeddingsIdentity_58:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_59AssignVariableOp2assignvariableop_59_adam_m_embedding_90_embeddingsIdentity_59:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_60AssignVariableOp2assignvariableop_60_adam_v_embedding_90_embeddingsIdentity_60:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_61AssignVariableOp2assignvariableop_61_adam_m_embedding_87_embeddingsIdentity_61:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_62AssignVariableOp2assignvariableop_62_adam_v_embedding_87_embeddingsIdentity_62:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_63AssignVariableOp2assignvariableop_63_adam_m_embedding_88_embeddingsIdentity_63:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_64AssignVariableOp2assignvariableop_64_adam_v_embedding_88_embeddingsIdentity_64:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:…
AssignVariableOp_65AssignVariableOp0assignvariableop_65_adam_m_user_embedding_kernelIdentity_65:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:…
AssignVariableOp_66AssignVariableOp0assignvariableop_66_adam_v_user_embedding_kernelIdentity_66:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_67AssignVariableOp.assignvariableop_67_adam_m_user_embedding_biasIdentity_67:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_68AssignVariableOp.assignvariableop_68_adam_v_user_embedding_biasIdentity_68:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:…
AssignVariableOp_69AssignVariableOp0assignvariableop_69_adam_m_food_embedding_kernelIdentity_69:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:…
AssignVariableOp_70AssignVariableOp0assignvariableop_70_adam_v_food_embedding_kernelIdentity_70:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_71AssignVariableOp.assignvariableop_71_adam_m_food_embedding_biasIdentity_71:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_72AssignVariableOp.assignvariableop_72_adam_v_food_embedding_biasIdentity_72:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:ћ
AssignVariableOp_73AssignVariableOp3assignvariableop_73_adam_m_context_embedding_kernelIdentity_73:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:ћ
AssignVariableOp_74AssignVariableOp3assignvariableop_74_adam_v_context_embedding_kernelIdentity_74:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_75AssignVariableOp1assignvariableop_75_adam_m_context_embedding_biasIdentity_75:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_76AssignVariableOp1assignvariableop_76_adam_v_context_embedding_biasIdentity_76:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:–
AssignVariableOp_77AssignVariableOp7assignvariableop_77_adam_m_batch_normalization_12_gammaIdentity_77:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:–
AssignVariableOp_78AssignVariableOp7assignvariableop_78_adam_v_batch_normalization_12_gammaIdentity_78:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:ѕ
AssignVariableOp_79AssignVariableOp6assignvariableop_79_adam_m_batch_normalization_12_betaIdentity_79:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:ѕ
AssignVariableOp_80AssignVariableOp6assignvariableop_80_adam_v_batch_normalization_12_betaIdentity_80:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:≈
AssignVariableOp_81AssignVariableOp,assignvariableop_81_adam_m_fc_layer_0_kernelIdentity_81:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:≈
AssignVariableOp_82AssignVariableOp,assignvariableop_82_adam_v_fc_layer_0_kernelIdentity_82:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:√
AssignVariableOp_83AssignVariableOp*assignvariableop_83_adam_m_fc_layer_0_biasIdentity_83:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:√
AssignVariableOp_84AssignVariableOp*assignvariableop_84_adam_v_fc_layer_0_biasIdentity_84:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:≈
AssignVariableOp_85AssignVariableOp,assignvariableop_85_adam_m_fc_layer_1_kernelIdentity_85:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:≈
AssignVariableOp_86AssignVariableOp,assignvariableop_86_adam_v_fc_layer_1_kernelIdentity_86:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:√
AssignVariableOp_87AssignVariableOp*assignvariableop_87_adam_m_fc_layer_1_biasIdentity_87:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:√
AssignVariableOp_88AssignVariableOp*assignvariableop_88_adam_v_fc_layer_1_biasIdentity_88:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:≈
AssignVariableOp_89AssignVariableOp,assignvariableop_89_adam_m_fc_layer_2_kernelIdentity_89:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:≈
AssignVariableOp_90AssignVariableOp,assignvariableop_90_adam_v_fc_layer_2_kernelIdentity_90:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:√
AssignVariableOp_91AssignVariableOp*assignvariableop_91_adam_m_fc_layer_2_biasIdentity_91:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:√
AssignVariableOp_92AssignVariableOp*assignvariableop_92_adam_v_fc_layer_2_biasIdentity_92:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:√
AssignVariableOp_93AssignVariableOp*assignvariableop_93_adam_m_output_0_kernelIdentity_93:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:√
AssignVariableOp_94AssignVariableOp*assignvariableop_94_adam_v_output_0_kernelIdentity_94:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_95AssignVariableOp(assignvariableop_95_adam_m_output_0_biasIdentity_95:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_96AssignVariableOp(assignvariableop_96_adam_v_output_0_biasIdentity_96:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:і
AssignVariableOp_97AssignVariableOpassignvariableop_97_total_1Identity_97:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:і
AssignVariableOp_98AssignVariableOpassignvariableop_98_count_1Identity_98:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:≤
AssignVariableOp_99AssignVariableOpassignvariableop_99_totalIdentity_99:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:µ
AssignVariableOp_100AssignVariableOpassignvariableop_100_countIdentity_100:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:ј
AssignVariableOp_101AssignVariableOp%assignvariableop_101_true_positives_1Identity_101:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:њ
AssignVariableOp_102AssignVariableOp$assignvariableop_102_false_positivesIdentity_102:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:Њ
AssignVariableOp_103AssignVariableOp#assignvariableop_103_true_positivesIdentity_103:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:њ
AssignVariableOp_104AssignVariableOp$assignvariableop_104_false_negativesIdentity_104:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 џ
Identity_105Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: Y
Identity_106IdentityIdentity_105:output:0^NoOp_1*
T0*
_output_shapes
: £
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*
_output_shapes
 "%
identity_106Identity_106:output:0*(
_construction_contextkEagerRuntime*й
_input_shapes„
‘: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042*
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
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:73
1
_user_specified_nameembedding_76/embeddings:73
1
_user_specified_nameembedding_77/embeddings:73
1
_user_specified_nameembedding_78/embeddings:73
1
_user_specified_nameembedding_79/embeddings:73
1
_user_specified_nameembedding_80/embeddings:73
1
_user_specified_nameembedding_81/embeddings:73
1
_user_specified_nameembedding_82/embeddings:73
1
_user_specified_nameembedding_83/embeddings:7	3
1
_user_specified_nameembedding_84/embeddings:7
3
1
_user_specified_nameembedding_85/embeddings:73
1
_user_specified_nameembedding_86/embeddings:73
1
_user_specified_nameembedding_89/embeddings:73
1
_user_specified_nameembedding_90/embeddings:73
1
_user_specified_nameembedding_87/embeddings:73
1
_user_specified_nameembedding_88/embeddings:51
/
_user_specified_nameuser_embedding/kernel:3/
-
_user_specified_nameuser_embedding/bias:51
/
_user_specified_namefood_embedding/kernel:3/
-
_user_specified_namefood_embedding/bias:84
2
_user_specified_namecontext_embedding/kernel:62
0
_user_specified_namecontext_embedding/bias:<8
6
_user_specified_namebatch_normalization_12/gamma:;7
5
_user_specified_namebatch_normalization_12/beta:B>
<
_user_specified_name$"batch_normalization_12/moving_mean:FB
@
_user_specified_name(&batch_normalization_12/moving_variance:1-
+
_user_specified_namefc_layer_0/kernel:/+
)
_user_specified_namefc_layer_0/bias:1-
+
_user_specified_namefc_layer_1/kernel:/+
)
_user_specified_namefc_layer_1/bias:1-
+
_user_specified_namefc_layer_2/kernel:/+
)
_user_specified_namefc_layer_2/bias:/ +
)
_user_specified_nameoutput_0/kernel:-!)
'
_user_specified_nameoutput_0/bias:)"%
#
_user_specified_name	iteration:-#)
'
_user_specified_namelearning_rate:>$:
8
_user_specified_name Adam/m/embedding_76/embeddings:>%:
8
_user_specified_name Adam/v/embedding_76/embeddings:>&:
8
_user_specified_name Adam/m/embedding_77/embeddings:>':
8
_user_specified_name Adam/v/embedding_77/embeddings:>(:
8
_user_specified_name Adam/m/embedding_78/embeddings:>):
8
_user_specified_name Adam/v/embedding_78/embeddings:>*:
8
_user_specified_name Adam/m/embedding_79/embeddings:>+:
8
_user_specified_name Adam/v/embedding_79/embeddings:>,:
8
_user_specified_name Adam/m/embedding_80/embeddings:>-:
8
_user_specified_name Adam/v/embedding_80/embeddings:>.:
8
_user_specified_name Adam/m/embedding_81/embeddings:>/:
8
_user_specified_name Adam/v/embedding_81/embeddings:>0:
8
_user_specified_name Adam/m/embedding_82/embeddings:>1:
8
_user_specified_name Adam/v/embedding_82/embeddings:>2:
8
_user_specified_name Adam/m/embedding_83/embeddings:>3:
8
_user_specified_name Adam/v/embedding_83/embeddings:>4:
8
_user_specified_name Adam/m/embedding_84/embeddings:>5:
8
_user_specified_name Adam/v/embedding_84/embeddings:>6:
8
_user_specified_name Adam/m/embedding_85/embeddings:>7:
8
_user_specified_name Adam/v/embedding_85/embeddings:>8:
8
_user_specified_name Adam/m/embedding_86/embeddings:>9:
8
_user_specified_name Adam/v/embedding_86/embeddings:>::
8
_user_specified_name Adam/m/embedding_89/embeddings:>;:
8
_user_specified_name Adam/v/embedding_89/embeddings:><:
8
_user_specified_name Adam/m/embedding_90/embeddings:>=:
8
_user_specified_name Adam/v/embedding_90/embeddings:>>:
8
_user_specified_name Adam/m/embedding_87/embeddings:>?:
8
_user_specified_name Adam/v/embedding_87/embeddings:>@:
8
_user_specified_name Adam/m/embedding_88/embeddings:>A:
8
_user_specified_name Adam/v/embedding_88/embeddings:<B8
6
_user_specified_nameAdam/m/user_embedding/kernel:<C8
6
_user_specified_nameAdam/v/user_embedding/kernel::D6
4
_user_specified_nameAdam/m/user_embedding/bias::E6
4
_user_specified_nameAdam/v/user_embedding/bias:<F8
6
_user_specified_nameAdam/m/food_embedding/kernel:<G8
6
_user_specified_nameAdam/v/food_embedding/kernel::H6
4
_user_specified_nameAdam/m/food_embedding/bias::I6
4
_user_specified_nameAdam/v/food_embedding/bias:?J;
9
_user_specified_name!Adam/m/context_embedding/kernel:?K;
9
_user_specified_name!Adam/v/context_embedding/kernel:=L9
7
_user_specified_nameAdam/m/context_embedding/bias:=M9
7
_user_specified_nameAdam/v/context_embedding/bias:CN?
=
_user_specified_name%#Adam/m/batch_normalization_12/gamma:CO?
=
_user_specified_name%#Adam/v/batch_normalization_12/gamma:BP>
<
_user_specified_name$"Adam/m/batch_normalization_12/beta:BQ>
<
_user_specified_name$"Adam/v/batch_normalization_12/beta:8R4
2
_user_specified_nameAdam/m/fc_layer_0/kernel:8S4
2
_user_specified_nameAdam/v/fc_layer_0/kernel:6T2
0
_user_specified_nameAdam/m/fc_layer_0/bias:6U2
0
_user_specified_nameAdam/v/fc_layer_0/bias:8V4
2
_user_specified_nameAdam/m/fc_layer_1/kernel:8W4
2
_user_specified_nameAdam/v/fc_layer_1/kernel:6X2
0
_user_specified_nameAdam/m/fc_layer_1/bias:6Y2
0
_user_specified_nameAdam/v/fc_layer_1/bias:8Z4
2
_user_specified_nameAdam/m/fc_layer_2/kernel:8[4
2
_user_specified_nameAdam/v/fc_layer_2/kernel:6\2
0
_user_specified_nameAdam/m/fc_layer_2/bias:6]2
0
_user_specified_nameAdam/v/fc_layer_2/bias:6^2
0
_user_specified_nameAdam/m/output_0/kernel:6_2
0
_user_specified_nameAdam/v/output_0/kernel:4`0
.
_user_specified_nameAdam/m/output_0/bias:4a0
.
_user_specified_nameAdam/v/output_0/bias:'b#
!
_user_specified_name	total_1:'c#
!
_user_specified_name	count_1:%d!

_user_specified_nametotal:%e!

_user_specified_namecount:0f,
*
_user_specified_nametrue_positives_1:/g+
)
_user_specified_namefalse_positives:.h*
(
_user_specified_nametrue_positives:/i+
)
_user_specified_namefalse_negatives
ѓ
<
__inference__creator_7097964
identityИҐ
hash_tableo

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	6434354*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: /
NoOpNoOp^hash_table*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
°
І
I__inference_embedding_87_layer_call_and_return_conditional_losses_7095259

inputs	*
embedding_lookup_7095254:
identityИҐembedding_lookupї
embedding_lookupResourceGatherembedding_lookup_7095254inputs*
Tindices0	*+
_class!
loc:@embedding_lookup/7095254*+
_output_shapes
:€€€€€€€€€*
dtype0v
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_output_shapes
:€€€€€€€€€u
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€5
NoOpNoOp^embedding_lookup*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:€€€€€€€€€: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:'#
!
_user_specified_name	7095254
Љ
С
K__inference_concatenate_34_layer_call_and_return_conditional_losses_7095562

inputs
inputs_1
inputs_2
inputs_3
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :К
concatConcatV2inputsinputs_1inputs_2inputs_3concat/axis:output:0*
N*
T0*(
_output_shapes
:€€€€€€€€€йX
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:€€€€€€€€€й"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:€€€€€€€€€ђ:€€€€€€€€€ђ:€€€€€€€€€:€€€€€€€€€:P L
(
_output_shapes
:€€€€€€€€€ђ
 
_user_specified_nameinputs:PL
(
_output_shapes
:€€€€€€€€€ђ
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
њ
c
G__inference_flatten_85_layer_call_and_return_conditional_losses_7097438

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:€€€€€€€€€X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
°
І
I__inference_embedding_90_layer_call_and_return_conditional_losses_7097328

inputs	*
embedding_lookup_7097323:W	
identityИҐembedding_lookupї
embedding_lookupResourceGatherembedding_lookup_7097323inputs*
Tindices0	*+
_class!
loc:@embedding_lookup/7097323*+
_output_shapes
:€€€€€€€€€	*
dtype0v
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_output_shapes
:€€€€€€€€€	u
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€	5
NoOpNoOp^embedding_lookup*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:€€€€€€€€€: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:'#
!
_user_specified_name	7097323
°
І
I__inference_embedding_80_layer_call_and_return_conditional_losses_7095193

inputs	*
embedding_lookup_7095188:
identityИҐembedding_lookupї
embedding_lookupResourceGatherembedding_lookup_7095188inputs*
Tindices0	*+
_class!
loc:@embedding_lookup/7095188*+
_output_shapes
:€€€€€€€€€*
dtype0v
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_output_shapes
:€€€€€€€€€u
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€5
NoOpNoOp^embedding_lookup*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:€€€€€€€€€: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:'#
!
_user_specified_name	7095188
•
е
K__inference_concatenate_33_layer_call_and_return_conditional_losses_7095436

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :∆
concatConcatV2inputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9concat/axis:output:0*
N
*
T0*(
_output_shapes
:€€€€€€€€€£X
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:€€€€€€€€€£"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*‘
_input_shapes¬
њ:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€	:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€А:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€	
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:P	L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
¬
Є
K__inference_user_embedding_layer_call_and_return_conditional_losses_7095472

inputs1
matmul_readvariableop_resource:	ђ.
biasadd_readvariableop_resource:	ђ
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐ7user_embedding/kernel/Regularizer/L2Loss/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђЧ
7user_embedding/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0Ф
(user_embedding/kernel/Regularizer/L2LossL2Loss?user_embedding/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: l
'user_embedding/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„£<≤
%user_embedding/kernel/Regularizer/mulMul0user_embedding/kernel/Regularizer/mul/x:output:01user_embedding/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: `
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€ђН
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp8^user_embedding/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2r
7user_embedding/kernel/Regularizer/L2Loss/ReadVariableOp7user_embedding/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Х
p
$__inference__update_step_xla_7097039
gradient

gradient_1	

gradient_2
variable:W	*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:€€€€€€€€€	:€€€€€€€€€:: *
	_noinline(:Q M
'
_output_shapes
:€€€€€€€€€	
"
_user_specified_name
gradient:MI
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
gradient:D@

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Х
p
$__inference__update_step_xla_7096962
gradient

gradient_1	

gradient_2
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:€€€€€€€€€:€€€€€€€€€:: *
	_noinline(:Q M
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
gradient:MI
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
gradient:D@

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
К
°
3__inference_context_embedding_layer_call_fn_7097665

inputs
unknown:	Ш
	unknown_0:
identityИҐStatefulPartitionedCallж
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_context_embedding_layer_call_and_return_conditional_losses_7095521o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€Ш: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€Ш
 
_user_specified_nameinputs:'#
!
_user_specified_name	7097659:'#
!
_user_specified_name	7097661
∞
M
$__inference__update_step_xla_7097088
gradient
variable:	й*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
	:й: *
	_noinline(:E A

_output_shapes	
:й
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
ѓ
Е
 __inference__initializer_7098106:
6key_value_init6434812_lookuptableimportv2_table_handle2
.key_value_init6434812_lookuptableimportv2_keys4
0key_value_init6434812_lookuptableimportv2_values	
identityИҐ)key_value_init6434812/LookupTableImportV2З
)key_value_init6434812/LookupTableImportV2LookupTableImportV26key_value_init6434812_lookuptableimportv2_table_handle.key_value_init6434812_lookuptableimportv2_keys0key_value_init6434812_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: N
NoOpNoOp*^key_value_init6434812/LookupTableImportV2*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2V
)key_value_init6434812/LookupTableImportV2)key_value_init6434812/LookupTableImportV2:, (
&
_user_specified_nametable_handle: 

_output_shapes
:: 

_output_shapes
:
µ	
„
8__inference_batch_normalization_12_layer_call_fn_7097741

inputs
unknown:	й
	unknown_0:	й
	unknown_1:	й
	unknown_2:	й
identityИҐStatefulPartitionedCallД
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€й*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_7094840p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€й<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€й: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€й
 
_user_specified_nameinputs:'#
!
_user_specified_name	7097731:'#
!
_user_specified_name	7097733:'#
!
_user_specified_name	7097735:'#
!
_user_specified_name	7097737
°
І
I__inference_embedding_82_layer_call_and_return_conditional_losses_7095171

inputs	*
embedding_lookup_7095166:
identityИҐembedding_lookupї
embedding_lookupResourceGatherembedding_lookup_7095166inputs*
Tindices0	*+
_class!
loc:@embedding_lookup/7095166*+
_output_shapes
:€€€€€€€€€*
dtype0v
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_output_shapes
:€€€€€€€€€u
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€5
NoOpNoOp^embedding_lookup*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:€€€€€€€€€: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:'#
!
_user_specified_name	7095166
ё
Ѓ
G__inference_fc_layer_2_layer_call_and_return_conditional_losses_7097880

inputs0
matmul_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐ3fc_layer_2/kernel/Regularizer/L2Loss/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€ Т
3fc_layer_2/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0М
$fc_layer_2/kernel/Regularizer/L2LossL2Loss;fc_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#fc_layer_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„£<¶
!fc_layer_2/kernel/Regularizer/mulMul,fc_layer_2/kernel/Regularizer/mul/x:output:0-fc_layer_2/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ Й
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp4^fc_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2j
3fc_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp3fc_layer_2/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
њ
c
G__inference_flatten_87_layer_call_and_return_conditional_losses_7095413

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:€€€€€€€€€X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
»
У
K__inference_concatenate_34_layer_call_and_return_conditional_losses_7097728
inputs_0
inputs_1
inputs_2
inputs_3
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :М
concatConcatV2inputs_0inputs_1inputs_2inputs_3concat/axis:output:0*
N*
T0*(
_output_shapes
:€€€€€€€€€йX
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:€€€€€€€€€й"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:€€€€€€€€€ђ:€€€€€€€€€ђ:€€€€€€€€€:€€€€€€€€€:R N
(
_output_shapes
:€€€€€€€€€ђ
"
_user_specified_name
inputs_0:RN
(
_output_shapes
:€€€€€€€€€ђ
"
_user_specified_name
inputs_1:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_2:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_3
°
І
I__inference_embedding_80_layer_call_and_return_conditional_losses_7097208

inputs	*
embedding_lookup_7097203:
identityИҐembedding_lookupї
embedding_lookupResourceGatherembedding_lookup_7097203inputs*
Tindices0	*+
_class!
loc:@embedding_lookup/7097203*+
_output_shapes
:€€€€€€€€€*
dtype0v
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_output_shapes
:€€€€€€€€€u
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€5
NoOpNoOp^embedding_lookup*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:€€€€€€€€€: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:'#
!
_user_specified_name	7097203
і
В
.__inference_embedding_88_layer_call_fn_7097493

inputs	
unknown:
identityИҐStatefulPartitionedCallЎ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_embedding_88_layer_call_and_return_conditional_losses_7095248s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:€€€€€€€€€: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:'#
!
_user_specified_name	7097489
њ
c
G__inference_flatten_85_layer_call_and_return_conditional_losses_7095372

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:€€€€€€€€€X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
њ
c
G__inference_flatten_82_layer_call_and_return_conditional_losses_7095351

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:€€€€€€€€€X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
њ
c
G__inference_flatten_83_layer_call_and_return_conditional_losses_7097416

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:€€€€€€€€€X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
µ
m
C__inference_dot_12_layer_call_and_return_conditional_losses_7095552

inputs
inputs_1
identityX
l2_normalize/SquareSquareinputs*
T0*(
_output_shapes
:€€€€€€€€€ђd
"l2_normalize/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :†
l2_normalize/SumSuml2_normalize/Square:y:0+l2_normalize/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
	keep_dims([
l2_normalize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *ћЉМ+Н
l2_normalize/MaximumMaximuml2_normalize/Sum:output:0l2_normalize/Maximum/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€g
l2_normalize/RsqrtRsqrtl2_normalize/Maximum:z:0*
T0*'
_output_shapes
:€€€€€€€€€f
l2_normalizeMulinputsl2_normalize/Rsqrt:y:0*
T0*(
_output_shapes
:€€€€€€€€€ђ\
l2_normalize_1/SquareSquareinputs_1*
T0*(
_output_shapes
:€€€€€€€€€ђf
$l2_normalize_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :¶
l2_normalize_1/SumSuml2_normalize_1/Square:y:0-l2_normalize_1/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
	keep_dims(]
l2_normalize_1/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *ћЉМ+У
l2_normalize_1/MaximumMaximuml2_normalize_1/Sum:output:0!l2_normalize_1/Maximum/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€k
l2_normalize_1/RsqrtRsqrtl2_normalize_1/Maximum:z:0*
T0*'
_output_shapes
:€€€€€€€€€l
l2_normalize_1Mulinputs_1l2_normalize_1/Rsqrt:y:0*
T0*(
_output_shapes
:€€€€€€€€€ђP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :z

ExpandDims
ExpandDimsl2_normalize:z:0ExpandDims/dim:output:0*
T0*,
_output_shapes
:€€€€€€€€€ђR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :А
ExpandDims_1
ExpandDimsl2_normalize_1:z:0ExpandDims_1/dim:output:0*
T0*,
_output_shapes
:€€€€€€€€€ђy
MatMulBatchMatMulV2ExpandDims:output:0ExpandDims_1:output:0*
T0*+
_output_shapes
:€€€€€€€€€R
ShapeShapeMatMul:output:0*
T0*
_output_shapes
::нѕl
SqueezeSqueezeMatMul:output:0*
T0*'
_output_shapes
:€€€€€€€€€*
squeeze_dims
X
IdentityIdentitySqueeze:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:€€€€€€€€€ђ:€€€€€€€€€ђ:P L
(
_output_shapes
:€€€€€€€€€ђ
 
_user_specified_nameinputs:PL
(
_output_shapes
:€€€€€€€€€ђ
 
_user_specified_nameinputs
њ
c
G__inference_flatten_76_layer_call_and_return_conditional_losses_7097339

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:€€€€€€€€€X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
≈
™
E__inference_output_0_layer_call_and_return_conditional_losses_7097904

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐ1output_0/kernel/Regularizer/L2Loss/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€Р
1output_0/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0И
"output_0/kernel/Regularizer/L2LossL2Loss9output_0/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!output_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„£<†
output_0/kernel/Regularizer/mulMul*output_0/kernel/Regularizer/mul/x:output:0+output_0/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€З
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp2^output_0/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2f
1output_0/kernel/Regularizer/L2Loss/ReadVariableOp1output_0/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
Ѓ
H
,__inference_flatten_80_layer_call_fn_7097377

inputs
identityµ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_flatten_80_layer_call_and_return_conditional_losses_7095337`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
њ
c
G__inference_flatten_77_layer_call_and_return_conditional_losses_7097350

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:€€€€€€€€€X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
°
І
I__inference_embedding_78_layer_call_and_return_conditional_losses_7097178

inputs	*
embedding_lookup_7097173:
identityИҐembedding_lookupї
embedding_lookupResourceGatherembedding_lookup_7097173inputs*
Tindices0	*+
_class!
loc:@embedding_lookup/7097173*+
_output_shapes
:€€€€€€€€€*
dtype0v
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_output_shapes
:€€€€€€€€€u
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€5
NoOpNoOp^embedding_lookup*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:€€€€€€€€€: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:'#
!
_user_specified_name	7097173
ґ
Я
K__inference_concatenate_32_layer_call_and_return_conditional_losses_7095506

inputs
inputs_1
inputs_2
inputs_3
inputs_4
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ф
concatConcatV2inputsinputs_1inputs_2inputs_3inputs_4concat/axis:output:0*
N*
T0*(
_output_shapes
:€€€€€€€€€ШX
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:€€€€€€€€€Ш"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*s
_input_shapesb
`:€€€€€€€€€:€€€€€€€€€У:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:PL
(
_output_shapes
:€€€€€€€€€У
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ь
.
__inference__destroyer_7098110
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
ѓ
Е
 __inference__initializer_7098016:
6key_value_init6434506_lookuptableimportv2_table_handle2
.key_value_init6434506_lookuptableimportv2_keys4
0key_value_init6434506_lookuptableimportv2_values	
identityИҐ)key_value_init6434506/LookupTableImportV2З
)key_value_init6434506/LookupTableImportV2LookupTableImportV26key_value_init6434506_lookuptableimportv2_table_handle.key_value_init6434506_lookuptableimportv2_keys0key_value_init6434506_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: N
NoOpNoOp*^key_value_init6434506/LookupTableImportV2*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2V
)key_value_init6434506/LookupTableImportV2)key_value_init6434506/LookupTableImportV2:, (
&
_user_specified_nametable_handle: 

_output_shapes
:: 

_output_shapes
:
Ь
.
__inference__destroyer_7098200
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
°
І
I__inference_embedding_83_layer_call_and_return_conditional_losses_7095160

inputs	*
embedding_lookup_7095155:
identityИҐembedding_lookupї
embedding_lookupResourceGatherembedding_lookup_7095155inputs*
Tindices0	*+
_class!
loc:@embedding_lookup/7095155*+
_output_shapes
:€€€€€€€€€*
dtype0v
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_output_shapes
:€€€€€€€€€u
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€5
NoOpNoOp^embedding_lookup*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:€€€€€€€€€: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:'#
!
_user_specified_name	7095155
Ј
x
0__inference_concatenate_34_layer_call_fn_7097719
inputs_0
inputs_1
inputs_2
inputs_3
identityЁ
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2inputs_3*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€й* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_concatenate_34_layer_call_and_return_conditional_losses_7095562a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:€€€€€€€€€й"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*a
_input_shapesP
N:€€€€€€€€€ђ:€€€€€€€€€ђ:€€€€€€€€€:€€€€€€€€€:R N
(
_output_shapes
:€€€€€€€€€ђ
"
_user_specified_name
inputs_0:RN
(
_output_shapes
:€€€€€€€€€ђ
"
_user_specified_name
inputs_1:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_2:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_3
Х
p
$__inference__update_step_xla_7096955
gradient

gradient_1	

gradient_2
variable:*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:€€€€€€€€€:€€€€€€€€€:: *
	_noinline(:Q M
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
gradient:MI
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
gradient:D@

_output_shapes
:
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
ѓ
<
__inference__creator_7097994
identityИҐ
hash_tableo

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	6434456*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: /
NoOpNoOp^hash_table*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
°
І
I__inference_embedding_85_layer_call_and_return_conditional_losses_7097283

inputs	*
embedding_lookup_7097278:
identityИҐembedding_lookupї
embedding_lookupResourceGatherembedding_lookup_7097278inputs*
Tindices0	*+
_class!
loc:@embedding_lookup/7097278*+
_output_shapes
:€€€€€€€€€*
dtype0v
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_output_shapes
:€€€€€€€€€u
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€5
NoOpNoOp^embedding_lookup*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:€€€€€€€€€: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:'#
!
_user_specified_name	7097278
∞
M
$__inference__update_step_xla_7097093
gradient
variable:	й*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
	:й: *
	_noinline(:E A

_output_shapes	
:й
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Ь
.
__inference__destroyer_7098155
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
њ
c
G__inference_flatten_78_layer_call_and_return_conditional_losses_7097361

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:€€€€€€€€€X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
°
І
I__inference_embedding_79_layer_call_and_return_conditional_losses_7097193

inputs	*
embedding_lookup_7097188:
identityИҐembedding_lookupї
embedding_lookupResourceGatherembedding_lookup_7097188inputs*
Tindices0	*+
_class!
loc:@embedding_lookup/7097188*+
_output_shapes
:€€€€€€€€€*
dtype0v
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_output_shapes
:€€€€€€€€€u
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€5
NoOpNoOp^embedding_lookup*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:€€€€€€€€€: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:'#
!
_user_specified_name	7097188
њ
c
G__inference_flatten_84_layer_call_and_return_conditional_losses_7095365

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:€€€€€€€€€X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
є
P
$__inference__update_step_xla_7097118
gradient
variable:@ *
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
:@ : *
	_noinline(:H D

_output_shapes

:@ 
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
Ь
.
__inference__destroyer_7098095
identityG
ConstConst*
_output_shapes
: *
dtype0*
value	B :E
IdentityIdentityConst:output:0*
T0*
_output_shapes
: "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 
µ	
Ж
0__inference_concatenate_32_layer_call_fn_7097646
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
identityи
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€Ш* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_concatenate_32_layer_call_and_return_conditional_losses_7095506a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:€€€€€€€€€Ш"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*s
_input_shapesb
`:€€€€€€€€€:€€€€€€€€€У:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:Q M
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_0:RN
(
_output_shapes
:€€€€€€€€€У
"
_user_specified_name
inputs_1:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_2:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_3:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_4
Ѓ
H
,__inference_flatten_86_layer_call_fn_7097443

inputs
identityµ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_flatten_86_layer_call_and_return_conditional_losses_7095379`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
°
І
I__inference_embedding_81_layer_call_and_return_conditional_losses_7097223

inputs	*
embedding_lookup_7097218:
identityИҐembedding_lookupї
embedding_lookupResourceGatherembedding_lookup_7097218inputs*
Tindices0	*+
_class!
loc:@embedding_lookup/7097218*+
_output_shapes
:€€€€€€€€€*
dtype0v
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_output_shapes
:€€€€€€€€€u
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€5
NoOpNoOp^embedding_lookup*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:€€€€€€€€€: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:'#
!
_user_specified_name	7097218
Ѓ
H
,__inference_flatten_87_layer_call_fn_7097574

inputs
identityµ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_flatten_87_layer_call_and_return_conditional_losses_7095413`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ѓ
Е
 __inference__initializer_7098166:
6key_value_init6435002_lookuptableimportv2_table_handle2
.key_value_init6435002_lookuptableimportv2_keys4
0key_value_init6435002_lookuptableimportv2_values	
identityИҐ)key_value_init6435002/LookupTableImportV2З
)key_value_init6435002/LookupTableImportV2LookupTableImportV26key_value_init6435002_lookuptableimportv2_table_handle.key_value_init6435002_lookuptableimportv2_keys0key_value_init6435002_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: N
NoOpNoOp*^key_value_init6435002/LookupTableImportV2*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2V
)key_value_init6435002/LookupTableImportV2)key_value_init6435002/LookupTableImportV2:, (
&
_user_specified_nametable_handle: 

_output_shapes
:: 

_output_shapes
:
°
І
I__inference_embedding_81_layer_call_and_return_conditional_losses_7095182

inputs	*
embedding_lookup_7095177:
identityИҐembedding_lookupї
embedding_lookupResourceGatherembedding_lookup_7095177inputs*
Tindices0	*+
_class!
loc:@embedding_lookup/7095177*+
_output_shapes
:€€€€€€€€€*
dtype0v
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_output_shapes
:€€€€€€€€€u
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€5
NoOpNoOp^embedding_lookup*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:€€€€€€€€€: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:'#
!
_user_specified_name	7095177
і
В
.__inference_embedding_79_layer_call_fn_7097185

inputs	
unknown:
identityИҐStatefulPartitionedCallЎ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:€€€€€€€€€*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_embedding_79_layer_call_and_return_conditional_losses_7095204s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:€€€€€€€€€: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:'#
!
_user_specified_name	7097181
Ѓ
H
,__inference_flatten_83_layer_call_fn_7097410

inputs
identityµ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_flatten_83_layer_call_and_return_conditional_losses_7095358`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
∞
M
$__inference__update_step_xla_7097103
gradient
variable:	А*
_XlaMustCompile(*(
_construction_contextkEagerRuntime*
_input_shapes
	:А: *
	_noinline(:E A

_output_shapes	
:А
"
_user_specified_name
gradient:($
"
_user_specified_name
variable
ѓ
<
__inference__creator_7098189
identityИҐ
hash_tableo

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	6435241*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: /
NoOpNoOp^hash_table*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
ѓ
<
__inference__creator_7097979
identityИҐ
hash_tableo

hash_tableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	6434405*
value_dtype0	W
IdentityIdentityhash_table:table_handle:0^NoOp*
T0*
_output_shapes
: /
NoOpNoOp^hash_table*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes 2

hash_table
hash_table
°
І
I__inference_embedding_87_layer_call_and_return_conditional_losses_7097486

inputs	*
embedding_lookup_7097481:
identityИҐembedding_lookupї
embedding_lookupResourceGatherembedding_lookup_7097481inputs*
Tindices0	*+
_class!
loc:@embedding_lookup/7097481*+
_output_shapes
:€€€€€€€€€*
dtype0v
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_output_shapes
:€€€€€€€€€u
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€5
NoOpNoOp^embedding_lookup*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:€€€€€€€€€: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:'#
!
_user_specified_name	7097481
µ&
р
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_7094840

inputs6
'assignmovingavg_readvariableop_resource:	й8
)assignmovingavg_1_readvariableop_resource:	й4
%batchnorm_mul_readvariableop_resource:	й0
!batchnorm_readvariableop_resource:	й
identityИҐAssignMovingAvgҐAssignMovingAvg/ReadVariableOpҐAssignMovingAvg_1Ґ AssignMovingAvg_1/ReadVariableOpҐbatchnorm/ReadVariableOpҐbatchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: А
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	й*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	йИ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:€€€€€€€€€йl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Я
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	й*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:й*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:й*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<Г
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:й*
dtype0В
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:йy
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:йђ
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
„#<З
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:й*
dtype0И
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:й
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:йі
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:йQ
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:й
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:й*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:йd
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:€€€€€€€€€йi
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:йw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:й*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:йs
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:€€€€€€€€€йc
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€й∆
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:€€€€€€€€€й: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€й
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
¬
Є
K__inference_user_embedding_layer_call_and_return_conditional_losses_7097614

inputs1
matmul_readvariableop_resource:	ђ.
biasadd_readvariableop_resource:	ђ
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐ7user_embedding/kernel/Regularizer/L2Loss/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€ђЧ
7user_embedding/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0Ф
(user_embedding/kernel/Regularizer/L2LossL2Loss?user_embedding/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: l
'user_embedding/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„£<≤
%user_embedding/kernel/Regularizer/mulMul0user_embedding/kernel/Regularizer/mul/x:output:01user_embedding/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: `
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€ђН
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp8^user_embedding/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2r
7user_embedding/kernel/Regularizer/L2Loss/ReadVariableOp7user_embedding/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
ѓ
Е
 __inference__initializer_7098091:
6key_value_init6434761_lookuptableimportv2_table_handle2
.key_value_init6434761_lookuptableimportv2_keys4
0key_value_init6434761_lookuptableimportv2_values	
identityИҐ)key_value_init6434761/LookupTableImportV2З
)key_value_init6434761/LookupTableImportV2LookupTableImportV26key_value_init6434761_lookuptableimportv2_table_handle.key_value_init6434761_lookuptableimportv2_keys0key_value_init6434761_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: N
NoOpNoOp*^key_value_init6434761/LookupTableImportV2*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2V
)key_value_init6434761/LookupTableImportV2)key_value_init6434761/LookupTableImportV2:, (
&
_user_specified_nametable_handle: 

_output_shapes
:: 

_output_shapes
:
г
љ
N__inference_context_embedding_layer_call_and_return_conditional_losses_7095521

inputs1
matmul_readvariableop_resource:	Ш-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐ:context_embedding/kernel/Regularizer/L2Loss/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ъ
:context_embedding/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Ш*
dtype0Ъ
+context_embedding/kernel/Regularizer/L2LossL2LossBcontext_embedding/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: o
*context_embedding/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„£<ї
(context_embedding/kernel/Regularizer/mulMul3context_embedding/kernel/Regularizer/mul/x:output:04context_embedding/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€Р
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp;^context_embedding/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€Ш: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2x
:context_embedding/kernel/Regularizer/L2Loss/ReadVariableOp:context_embedding/kernel/Regularizer/L2Loss/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€Ш
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
ѓ
Е
 __inference__initializer_7097986:
6key_value_init6434404_lookuptableimportv2_table_handle2
.key_value_init6434404_lookuptableimportv2_keys4
0key_value_init6434404_lookuptableimportv2_values	
identityИҐ)key_value_init6434404/LookupTableImportV2З
)key_value_init6434404/LookupTableImportV2LookupTableImportV26key_value_init6434404_lookuptableimportv2_table_handle.key_value_init6434404_lookuptableimportv2_keys0key_value_init6434404_lookuptableimportv2_values*	
Tin0*

Tout0	*
_output_shapes
 G
ConstConst*
_output_shapes
: *
dtype0*
value	B :L
IdentityIdentityConst:output:0^NoOp*
T0*
_output_shapes
: N
NoOpNoOp*^key_value_init6434404/LookupTableImportV2*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: ::2V
)key_value_init6434404/LookupTableImportV2)key_value_init6434404/LookupTableImportV2:, (
&
_user_specified_nametable_handle: 

_output_shapes
:: 

_output_shapes
:
І	
≤
__inference_loss_fn_6_7097960L
:output_0_kernel_regularizer_l2loss_readvariableop_resource: 
identityИҐ1output_0/kernel/Regularizer/L2Loss/ReadVariableOpђ
1output_0/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp:output_0_kernel_regularizer_l2loss_readvariableop_resource*
_output_shapes

: *
dtype0И
"output_0/kernel/Regularizer/L2LossL2Loss9output_0/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: f
!output_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„£<†
output_0/kernel/Regularizer/mulMul*output_0/kernel/Regularizer/mul/x:output:0+output_0/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentity#output_0/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: V
NoOpNoOp2^output_0/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2f
1output_0/kernel/Regularizer/L2Loss/ReadVariableOp1output_0/kernel/Regularizer/L2Loss/ReadVariableOp:( $
"
_user_specified_name
resource
≈	
Ј
__inference_loss_fn_4_7097944O
<fc_layer_1_kernel_regularizer_l2loss_readvariableop_resource:	А@
identityИҐ3fc_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp±
3fc_layer_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp<fc_layer_1_kernel_regularizer_l2loss_readvariableop_resource*
_output_shapes
:	А@*
dtype0М
$fc_layer_1/kernel/Regularizer/L2LossL2Loss;fc_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#fc_layer_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„£<¶
!fc_layer_1/kernel/Regularizer/mulMul,fc_layer_1/kernel/Regularizer/mul/x:output:0-fc_layer_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: c
IdentityIdentity%fc_layer_1/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: X
NoOpNoOp4^fc_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2j
3fc_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp3fc_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp:( $
"
_user_specified_name
resource
Ч
∞
K__inference_concatenate_31_layer_call_and_return_conditional_losses_7095457

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
	inputs_10
	inputs_11
	inputs_12
	inputs_13
	inputs_14
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ь
concatConcatV2inputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11	inputs_12	inputs_13	inputs_14concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*≤
_input_shapes†
Э:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:O	K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:O
K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:OK
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
њ
c
G__inference_flatten_88_layer_call_and_return_conditional_losses_7095420

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:€€€€€€€€€X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€:S O
+
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
°
І
I__inference_embedding_76_layer_call_and_return_conditional_losses_7095237

inputs	*
embedding_lookup_7095232:
identityИҐembedding_lookupї
embedding_lookupResourceGatherembedding_lookup_7095232inputs*
Tindices0	*+
_class!
loc:@embedding_lookup/7095232*+
_output_shapes
:€€€€€€€€€*
dtype0v
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_output_shapes
:€€€€€€€€€u
IdentityIdentity"embedding_lookup/Identity:output:0^NoOp*
T0*+
_output_shapes
:€€€€€€€€€5
NoOpNoOp^embedding_lookup*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:€€€€€€€€€: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:'#
!
_user_specified_name	7095232
€	
ј
__inference_loss_fn_1_7097920T
@food_embedding_kernel_regularizer_l2loss_readvariableop_resource:
£ђ
identityИҐ7food_embedding/kernel/Regularizer/L2Loss/ReadVariableOpЇ
7food_embedding/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp@food_embedding_kernel_regularizer_l2loss_readvariableop_resource* 
_output_shapes
:
£ђ*
dtype0Ф
(food_embedding/kernel/Regularizer/L2LossL2Loss?food_embedding/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: l
'food_embedding/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„£<≤
%food_embedding/kernel/Regularizer/mulMul0food_embedding/kernel/Regularizer/mul/x:output:01food_embedding/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: g
IdentityIdentity)food_embedding/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: \
NoOpNoOp8^food_embedding/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2r
7food_embedding/kernel/Regularizer/L2Loss/ReadVariableOp7food_embedding/kernel/Regularizer/L2Loss/ReadVariableOp:( $
"
_user_specified_name
resource
г
ѓ
G__inference_fc_layer_1_layer_call_and_return_conditional_losses_7097856

inputs1
matmul_readvariableop_resource:	А@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpҐ3fc_layer_1/kernel/Regularizer/L2Loss/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@У
3fc_layer_1/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А@*
dtype0М
$fc_layer_1/kernel/Regularizer/L2LossL2Loss;fc_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: h
#fc_layer_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
„£<¶
!fc_layer_1/kernel/Regularizer/mulMul,fc_layer_1/kernel/Regularizer/mul/x:output:0-fc_layer_1/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@Й
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp4^fc_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2j
3fc_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp3fc_layer_1/kernel/Regularizer/L2Loss/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource
≥
ћ
0__inference_concatenate_33_layer_call_fn_7097554
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
identityЯ
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9*
Tin
2
*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€£* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_concatenate_33_layer_call_and_return_conditional_losses_7095436a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:€€€€€€€€€£"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*‘
_input_shapes¬
њ:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€	:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€А:Q M
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_0:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_1:QM
'
_output_shapes
:€€€€€€€€€	
"
_user_specified_name
inputs_2:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_3:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_4:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_5:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_6:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_7:QM
'
_output_shapes
:€€€€€€€€€
"
_user_specified_name
inputs_8:R	N
(
_output_shapes
:€€€€€€€€€А
"
_user_specified_name
inputs_9"нN
saver_filename:0StatefulPartitionedCall_18:0StatefulPartitionedCall_198"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*я
serving_defaultЋ
3
BMI,
serving_default_BMI:0€€€€€€€€€
?
	age_range2
serving_default_age_range:0€€€€€€€€€
?
	allergens2
serving_default_allergens:0€€€€€€€€€
;
allergy0
serving_default_allergy:0€€€€€€€€€
=
calories1
serving_default_calories:0€€€€€€€€€
G
carbohydrates6
serving_default_carbohydrates:0€€€€€€€€€
K
clinical_gender8
!serving_default_clinical_gender:0€€€€€€€€€
K
cultural_factor8
!serving_default_cultural_factor:0€€€€€€€€€
U
cultural_restriction=
&serving_default_cultural_restriction:0€€€€€€€€€
Y
current_daily_calories?
(serving_default_current_daily_calories:0€€€€€€€€€
Y
current_working_status?
(serving_default_current_working_status:0€€€€€€€€€
A

day_number3
serving_default_day_number:0€€€€€€€€€
B

embeddings4
serving_default_embeddings:0€€€€€€€€€А
?
	ethnicity2
serving_default_ethnicity:0€€€€€€€€€
3
fat,
serving_default_fat:0€€€€€€€€€
7
fiber.
serving_default_fiber:0€€€€€€€€€
9
height/
serving_default_height:0€€€€€€€€€
A

life_style3
serving_default_life_style:0€€€€€€€€€
I
marital_status7
 serving_default_marital_status:0€€€€€€€€€
C
meal_type_y4
serving_default_meal_type_y:0€€€€€€€€€
=
next_BMI1
serving_default_next_BMI:0€€€€€€€€€
I
nutrition_goal7
 serving_default_nutrition_goal:0€€€€€€€€€
_
place_of_meal_consumptionB
+serving_default_place_of_meal_consumption:0€€€€€€€€€
7
price.
serving_default_price:0€€€€€€€€€
]
projected_daily_caloriesA
*serving_default_projected_daily_calories:0€€€€€€€€€
;
protein0
serving_default_protein:0€€€€€€€€€
u
$social_situation_of_meal_consumptionM
6serving_default_social_situation_of_meal_consumption:0€€€€€€€€€
7
taste.
serving_default_taste:0€€€€€€€€€
]
time_of_meal_consumptionA
*serving_default_time_of_meal_consumption:0€€€€€€€€€
9
weight/
serving_default_weight:0€€€€€€€€€<
output_00
StatefulPartitionedCall:0€€€€€€€€€tensorflow/serving/predict:Ќв

®
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer-14
layer-15
layer-16
layer-17
layer-18
layer-19
layer-20
layer-21
layer-22
layer-23
layer-24
layer-25
layer-26
layer-27
layer_with_weights-0
layer-28
layer_with_weights-1
layer-29
layer_with_weights-2
layer-30
 layer_with_weights-3
 layer-31
!layer_with_weights-4
!layer-32
"layer_with_weights-5
"layer-33
#layer_with_weights-6
#layer-34
$layer_with_weights-7
$layer-35
%layer_with_weights-8
%layer-36
&layer_with_weights-9
&layer-37
'layer_with_weights-10
'layer-38
(layer_with_weights-11
(layer-39
)layer_with_weights-12
)layer-40
*layer-41
+layer-42
,layer-43
-layer-44
.layer-45
/layer-46
0layer-47
1layer-48
2layer-49
3layer-50
4layer-51
5layer-52
6layer-53
7layer-54
8layer-55
9layer-56
:layer-57
;layer-58
<layer-59
=layer-60
>layer-61
?layer-62
@layer-63
Alayer-64
Blayer-65
Clayer-66
Dlayer-67
Elayer-68
Flayer-69
Glayer_with_weights-13
Glayer-70
Hlayer_with_weights-14
Hlayer-71
Ilayer-72
Jlayer-73
Klayer-74
Llayer-75
Mlayer-76
Nlayer-77
Olayer-78
Player_with_weights-15
Player-79
Qlayer_with_weights-16
Qlayer-80
Rlayer-81
Slayer_with_weights-17
Slayer-82
Tlayer-83
Ulayer-84
Vlayer_with_weights-18
Vlayer-85
Wlayer_with_weights-19
Wlayer-86
Xlayer_with_weights-20
Xlayer-87
Ylayer_with_weights-21
Ylayer-88
Zlayer_with_weights-22
Zlayer-89
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses
a_default_save_signature
b	optimizer
c
signatures"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
P
d	keras_api
einput_vocabulary
flookup_table"
_tf_keras_layer
P
g	keras_api
hinput_vocabulary
ilookup_table"
_tf_keras_layer
P
j	keras_api
kinput_vocabulary
llookup_table"
_tf_keras_layer
P
m	keras_api
ninput_vocabulary
olookup_table"
_tf_keras_layer
P
p	keras_api
qinput_vocabulary
rlookup_table"
_tf_keras_layer
P
s	keras_api
tinput_vocabulary
ulookup_table"
_tf_keras_layer
P
v	keras_api
winput_vocabulary
xlookup_table"
_tf_keras_layer
P
y	keras_api
zinput_vocabulary
{lookup_table"
_tf_keras_layer
P
|	keras_api
}input_vocabulary
~lookup_table"
_tf_keras_layer
R
	keras_api
Аinput_vocabulary
Бlookup_table"
_tf_keras_layer
S
В	keras_api
Гinput_vocabulary
Дlookup_table"
_tf_keras_layer
S
Е	keras_api
Жinput_vocabulary
Зlookup_table"
_tf_keras_layer
S
И	keras_api
Йinput_vocabulary
Кlookup_table"
_tf_keras_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
Љ
Л	variables
Мtrainable_variables
Нregularization_losses
О	keras_api
П__call__
+Р&call_and_return_all_conditional_losses
С
embeddings"
_tf_keras_layer
Љ
Т	variables
Уtrainable_variables
Фregularization_losses
Х	keras_api
Ц__call__
+Ч&call_and_return_all_conditional_losses
Ш
embeddings"
_tf_keras_layer
Љ
Щ	variables
Ъtrainable_variables
Ыregularization_losses
Ь	keras_api
Э__call__
+Ю&call_and_return_all_conditional_losses
Я
embeddings"
_tf_keras_layer
Љ
†	variables
°trainable_variables
Ґregularization_losses
£	keras_api
§__call__
+•&call_and_return_all_conditional_losses
¶
embeddings"
_tf_keras_layer
Љ
І	variables
®trainable_variables
©regularization_losses
™	keras_api
Ђ__call__
+ђ&call_and_return_all_conditional_losses
≠
embeddings"
_tf_keras_layer
Љ
Ѓ	variables
ѓtrainable_variables
∞regularization_losses
±	keras_api
≤__call__
+≥&call_and_return_all_conditional_losses
і
embeddings"
_tf_keras_layer
Љ
µ	variables
ґtrainable_variables
Јregularization_losses
Є	keras_api
є__call__
+Ї&call_and_return_all_conditional_losses
ї
embeddings"
_tf_keras_layer
Љ
Љ	variables
љtrainable_variables
Њregularization_losses
њ	keras_api
ј__call__
+Ѕ&call_and_return_all_conditional_losses
¬
embeddings"
_tf_keras_layer
Љ
√	variables
ƒtrainable_variables
≈regularization_losses
∆	keras_api
«__call__
+»&call_and_return_all_conditional_losses
…
embeddings"
_tf_keras_layer
Љ
 	variables
Ћtrainable_variables
ћregularization_losses
Ќ	keras_api
ќ__call__
+ѕ&call_and_return_all_conditional_losses
–
embeddings"
_tf_keras_layer
Љ
—	variables
“trainable_variables
”regularization_losses
‘	keras_api
’__call__
+÷&call_and_return_all_conditional_losses
„
embeddings"
_tf_keras_layer
Љ
Ў	variables
ўtrainable_variables
Џregularization_losses
џ	keras_api
№__call__
+Ё&call_and_return_all_conditional_losses
ё
embeddings"
_tf_keras_layer
Љ
я	variables
аtrainable_variables
бregularization_losses
в	keras_api
г__call__
+д&call_and_return_all_conditional_losses
е
embeddings"
_tf_keras_layer
"
_tf_keras_input_layer
S
ж	keras_api
зinput_vocabulary
иlookup_table"
_tf_keras_layer
S
й	keras_api
кinput_vocabulary
лlookup_table"
_tf_keras_layer
Ђ
м	variables
нtrainable_variables
оregularization_losses
п	keras_api
р__call__
+с&call_and_return_all_conditional_losses"
_tf_keras_layer
Ђ
т	variables
уtrainable_variables
фregularization_losses
х	keras_api
ц__call__
+ч&call_and_return_all_conditional_losses"
_tf_keras_layer
Ђ
ш	variables
щtrainable_variables
ъregularization_losses
ы	keras_api
ь__call__
+э&call_and_return_all_conditional_losses"
_tf_keras_layer
Ђ
ю	variables
€trainable_variables
Аregularization_losses
Б	keras_api
В__call__
+Г&call_and_return_all_conditional_losses"
_tf_keras_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
Ђ
Д	variables
Еtrainable_variables
Жregularization_losses
З	keras_api
И__call__
+Й&call_and_return_all_conditional_losses"
_tf_keras_layer
Ђ
К	variables
Лtrainable_variables
Мregularization_losses
Н	keras_api
О__call__
+П&call_and_return_all_conditional_losses"
_tf_keras_layer
Ђ
Р	variables
Сtrainable_variables
Тregularization_losses
У	keras_api
Ф__call__
+Х&call_and_return_all_conditional_losses"
_tf_keras_layer
Ђ
Ц	variables
Чtrainable_variables
Шregularization_losses
Щ	keras_api
Ъ__call__
+Ы&call_and_return_all_conditional_losses"
_tf_keras_layer
Ђ
Ь	variables
Эtrainable_variables
Юregularization_losses
Я	keras_api
†__call__
+°&call_and_return_all_conditional_losses"
_tf_keras_layer
Ђ
Ґ	variables
£trainable_variables
§regularization_losses
•	keras_api
¶__call__
+І&call_and_return_all_conditional_losses"
_tf_keras_layer
Ђ
®	variables
©trainable_variables
™regularization_losses
Ђ	keras_api
ђ__call__
+≠&call_and_return_all_conditional_losses"
_tf_keras_layer
Ђ
Ѓ	variables
ѓtrainable_variables
∞regularization_losses
±	keras_api
≤__call__
+≥&call_and_return_all_conditional_losses"
_tf_keras_layer
"
_tf_keras_input_layer
Ђ
і	variables
µtrainable_variables
ґregularization_losses
Ј	keras_api
Є__call__
+є&call_and_return_all_conditional_losses"
_tf_keras_layer
S
Ї	keras_api
їinput_vocabulary
Љlookup_table"
_tf_keras_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
Љ
љ	variables
Њtrainable_variables
њregularization_losses
ј	keras_api
Ѕ__call__
+¬&call_and_return_all_conditional_losses
√
embeddings"
_tf_keras_layer
Љ
ƒ	variables
≈trainable_variables
∆regularization_losses
«	keras_api
»__call__
+…&call_and_return_all_conditional_losses
 
embeddings"
_tf_keras_layer
Ђ
Ћ	variables
ћtrainable_variables
Ќregularization_losses
ќ	keras_api
ѕ__call__
+–&call_and_return_all_conditional_losses"
_tf_keras_layer
Ђ
—	variables
“trainable_variables
”regularization_losses
‘	keras_api
’__call__
+÷&call_and_return_all_conditional_losses"
_tf_keras_layer
"
_tf_keras_input_layer
S
„	keras_api
Ўinput_vocabulary
ўlookup_table"
_tf_keras_layer
"
_tf_keras_input_layer
Ђ
Џ	variables
џtrainable_variables
№regularization_losses
Ё	keras_api
ё__call__
+я&call_and_return_all_conditional_losses"
_tf_keras_layer
Ђ
а	variables
бtrainable_variables
вregularization_losses
г	keras_api
д__call__
+е&call_and_return_all_conditional_losses"
_tf_keras_layer
√
ж	variables
зtrainable_variables
иregularization_losses
й	keras_api
к__call__
+л&call_and_return_all_conditional_losses
мkernel
	нbias"
_tf_keras_layer
√
о	variables
пtrainable_variables
рregularization_losses
с	keras_api
т__call__
+у&call_and_return_all_conditional_losses
фkernel
	хbias"
_tf_keras_layer
Ђ
ц	variables
чtrainable_variables
шregularization_losses
щ	keras_api
ъ__call__
+ы&call_and_return_all_conditional_losses"
_tf_keras_layer
√
ь	variables
эtrainable_variables
юregularization_losses
€	keras_api
А__call__
+Б&call_and_return_all_conditional_losses
Вkernel
	Гbias"
_tf_keras_layer
Ђ
Д	variables
Еtrainable_variables
Жregularization_losses
З	keras_api
И__call__
+Й&call_and_return_all_conditional_losses"
_tf_keras_layer
Ђ
К	variables
Лtrainable_variables
Мregularization_losses
Н	keras_api
О__call__
+П&call_and_return_all_conditional_losses"
_tf_keras_layer
х
Р	variables
Сtrainable_variables
Тregularization_losses
У	keras_api
Ф__call__
+Х&call_and_return_all_conditional_losses
	Цaxis

Чgamma
	Шbeta
Щmoving_mean
Ъmoving_variance"
_tf_keras_layer
√
Ы	variables
Ьtrainable_variables
Эregularization_losses
Ю	keras_api
Я__call__
+†&call_and_return_all_conditional_losses
°kernel
	Ґbias"
_tf_keras_layer
√
£	variables
§trainable_variables
•regularization_losses
¶	keras_api
І__call__
+®&call_and_return_all_conditional_losses
©kernel
	™bias"
_tf_keras_layer
√
Ђ	variables
ђtrainable_variables
≠regularization_losses
Ѓ	keras_api
ѓ__call__
+∞&call_and_return_all_conditional_losses
±kernel
	≤bias"
_tf_keras_layer
√
≥	variables
іtrainable_variables
µregularization_losses
ґ	keras_api
Ј__call__
+Є&call_and_return_all_conditional_losses
єkernel
	Їbias"
_tf_keras_layer
њ
С0
Ш1
Я2
¶3
≠4
і5
ї6
¬7
…8
–9
„10
ё11
е12
√13
 14
м15
н16
ф17
х18
В19
Г20
Ч21
Ш22
Щ23
Ъ24
°25
Ґ26
©27
™28
±29
≤30
є31
Ї32"
trackable_list_wrapper
≠
С0
Ш1
Я2
¶3
≠4
і5
ї6
¬7
…8
–9
„10
ё11
е12
√13
 14
м15
н16
ф17
х18
В19
Г20
Ч21
Ш22
°23
Ґ24
©25
™26
±27
≤28
є29
Ї30"
trackable_list_wrapper
X
ї0
Љ1
љ2
Њ3
њ4
ј5
Ѕ6"
trackable_list_wrapper
ѕ
¬non_trainable_variables
√layers
ƒmetrics
 ≈layer_regularization_losses
∆layer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
a_default_save_signature
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
Ћ
«trace_0
»trace_12Р
*__inference_model_12_layer_call_fn_7096254
*__inference_model_12_layer_call_fn_7096422µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z«trace_0z»trace_1
Б
…trace_0
 trace_12∆
E__inference_model_12_layer_call_and_return_conditional_losses_7095682
E__inference_model_12_layer_call_and_return_conditional_losses_7096086µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z…trace_0z trace_1
ѓ	
Ћ	capture_1
ћ	capture_3
Ќ	capture_5
ќ	capture_7
ѕ	capture_9
–
capture_11
—
capture_13
“
capture_15
”
capture_17
‘
capture_19
’
capture_21
÷
capture_23
„
capture_25
Ў
capture_27
ў
capture_29
Џ
capture_46
џ
capture_48Bф
"__inference__wrapped_model_7094806BMI	age_range	allergensallergycaloriescarbohydratesclinical_gendercultural_factorcultural_restrictioncurrent_daily_caloriescurrent_working_status
day_number
embeddings	ethnicityfatfiberheight
life_stylemarital_statusmeal_type_ynext_BMInutrition_goalplace_of_meal_consumptionpriceprojected_daily_caloriesprotein$social_situation_of_meal_consumptiontastetime_of_meal_consumptionweight"Ш
С≤Н
FullArgSpec
argsЪ

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЋ	capture_1zћ	capture_3zЌ	capture_5zќ	capture_7zѕ	capture_9z–
capture_11z—
capture_13z“
capture_15z”
capture_17z‘
capture_19z’
capture_21z÷
capture_23z„
capture_25zЎ
capture_27zў
capture_29zЏ
capture_46zџ
capture_48
£
№
_variables
Ё_iterations
ё_learning_rate
я_index_dict
а
_momentums
б_velocities
в_update_step_xla"
experimentalOptimizer
-
гserving_default"
signature_map
"
_generic_user_object
 "
trackable_list_wrapper
j
д_initializer
е_create_resource
ж_initialize
з_destroy_resourceR jtf.StaticHashTable
"
_generic_user_object
 "
trackable_list_wrapper
j
и_initializer
й_create_resource
к_initialize
л_destroy_resourceR jtf.StaticHashTable
"
_generic_user_object
 "
trackable_list_wrapper
j
м_initializer
н_create_resource
о_initialize
п_destroy_resourceR jtf.StaticHashTable
"
_generic_user_object
 "
trackable_list_wrapper
j
р_initializer
с_create_resource
т_initialize
у_destroy_resourceR jtf.StaticHashTable
"
_generic_user_object
 "
trackable_list_wrapper
j
ф_initializer
х_create_resource
ц_initialize
ч_destroy_resourceR jtf.StaticHashTable
"
_generic_user_object
 "
trackable_list_wrapper
j
ш_initializer
щ_create_resource
ъ_initialize
ы_destroy_resourceR jtf.StaticHashTable
"
_generic_user_object
 "
trackable_list_wrapper
j
ь_initializer
э_create_resource
ю_initialize
€_destroy_resourceR jtf.StaticHashTable
"
_generic_user_object
 "
trackable_list_wrapper
j
А_initializer
Б_create_resource
В_initialize
Г_destroy_resourceR jtf.StaticHashTable
"
_generic_user_object
 "
trackable_list_wrapper
j
Д_initializer
Е_create_resource
Ж_initialize
З_destroy_resourceR jtf.StaticHashTable
"
_generic_user_object
 "
trackable_list_wrapper
j
И_initializer
Й_create_resource
К_initialize
Л_destroy_resourceR jtf.StaticHashTable
"
_generic_user_object
 "
trackable_list_wrapper
j
М_initializer
Н_create_resource
О_initialize
П_destroy_resourceR jtf.StaticHashTable
"
_generic_user_object
 "
trackable_list_wrapper
j
Р_initializer
С_create_resource
Т_initialize
У_destroy_resourceR jtf.StaticHashTable
"
_generic_user_object
 "
trackable_list_wrapper
j
Ф_initializer
Х_create_resource
Ц_initialize
Ч_destroy_resourceR jtf.StaticHashTable
(
С0"
trackable_list_wrapper
(
С0"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Шnon_trainable_variables
Щlayers
Ъmetrics
 Ыlayer_regularization_losses
Ьlayer_metrics
Л	variables
Мtrainable_variables
Нregularization_losses
П__call__
+Р&call_and_return_all_conditional_losses
'Р"call_and_return_conditional_losses"
_generic_user_object
к
Эtrace_02Ћ
.__inference_embedding_76_layer_call_fn_7097140Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЭtrace_0
Е
Юtrace_02ж
I__inference_embedding_76_layer_call_and_return_conditional_losses_7097148Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЮtrace_0
):'2embedding_76/embeddings
(
Ш0"
trackable_list_wrapper
(
Ш0"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Яnon_trainable_variables
†layers
°metrics
 Ґlayer_regularization_losses
£layer_metrics
Т	variables
Уtrainable_variables
Фregularization_losses
Ц__call__
+Ч&call_and_return_all_conditional_losses
'Ч"call_and_return_conditional_losses"
_generic_user_object
к
§trace_02Ћ
.__inference_embedding_77_layer_call_fn_7097155Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z§trace_0
Е
•trace_02ж
I__inference_embedding_77_layer_call_and_return_conditional_losses_7097163Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z•trace_0
):'2embedding_77/embeddings
(
Я0"
trackable_list_wrapper
(
Я0"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
¶non_trainable_variables
Іlayers
®metrics
 ©layer_regularization_losses
™layer_metrics
Щ	variables
Ъtrainable_variables
Ыregularization_losses
Э__call__
+Ю&call_and_return_all_conditional_losses
'Ю"call_and_return_conditional_losses"
_generic_user_object
к
Ђtrace_02Ћ
.__inference_embedding_78_layer_call_fn_7097170Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЂtrace_0
Е
ђtrace_02ж
I__inference_embedding_78_layer_call_and_return_conditional_losses_7097178Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zђtrace_0
):'2embedding_78/embeddings
(
¶0"
trackable_list_wrapper
(
¶0"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
≠non_trainable_variables
Ѓlayers
ѓmetrics
 ∞layer_regularization_losses
±layer_metrics
†	variables
°trainable_variables
Ґregularization_losses
§__call__
+•&call_and_return_all_conditional_losses
'•"call_and_return_conditional_losses"
_generic_user_object
к
≤trace_02Ћ
.__inference_embedding_79_layer_call_fn_7097185Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z≤trace_0
Е
≥trace_02ж
I__inference_embedding_79_layer_call_and_return_conditional_losses_7097193Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z≥trace_0
):'2embedding_79/embeddings
(
≠0"
trackable_list_wrapper
(
≠0"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
іnon_trainable_variables
µlayers
ґmetrics
 Јlayer_regularization_losses
Єlayer_metrics
І	variables
®trainable_variables
©regularization_losses
Ђ__call__
+ђ&call_and_return_all_conditional_losses
'ђ"call_and_return_conditional_losses"
_generic_user_object
к
єtrace_02Ћ
.__inference_embedding_80_layer_call_fn_7097200Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zєtrace_0
Е
Їtrace_02ж
I__inference_embedding_80_layer_call_and_return_conditional_losses_7097208Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЇtrace_0
):'2embedding_80/embeddings
(
і0"
trackable_list_wrapper
(
і0"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
їnon_trainable_variables
Љlayers
љmetrics
 Њlayer_regularization_losses
њlayer_metrics
Ѓ	variables
ѓtrainable_variables
∞regularization_losses
≤__call__
+≥&call_and_return_all_conditional_losses
'≥"call_and_return_conditional_losses"
_generic_user_object
к
јtrace_02Ћ
.__inference_embedding_81_layer_call_fn_7097215Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zјtrace_0
Е
Ѕtrace_02ж
I__inference_embedding_81_layer_call_and_return_conditional_losses_7097223Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЅtrace_0
):'2embedding_81/embeddings
(
ї0"
trackable_list_wrapper
(
ї0"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
¬non_trainable_variables
√layers
ƒmetrics
 ≈layer_regularization_losses
∆layer_metrics
µ	variables
ґtrainable_variables
Јregularization_losses
є__call__
+Ї&call_and_return_all_conditional_losses
'Ї"call_and_return_conditional_losses"
_generic_user_object
к
«trace_02Ћ
.__inference_embedding_82_layer_call_fn_7097230Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z«trace_0
Е
»trace_02ж
I__inference_embedding_82_layer_call_and_return_conditional_losses_7097238Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z»trace_0
):'2embedding_82/embeddings
(
¬0"
trackable_list_wrapper
(
¬0"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
…non_trainable_variables
 layers
Ћmetrics
 ћlayer_regularization_losses
Ќlayer_metrics
Љ	variables
љtrainable_variables
Њregularization_losses
ј__call__
+Ѕ&call_and_return_all_conditional_losses
'Ѕ"call_and_return_conditional_losses"
_generic_user_object
к
ќtrace_02Ћ
.__inference_embedding_83_layer_call_fn_7097245Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zќtrace_0
Е
ѕtrace_02ж
I__inference_embedding_83_layer_call_and_return_conditional_losses_7097253Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zѕtrace_0
):'2embedding_83/embeddings
(
…0"
trackable_list_wrapper
(
…0"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
–non_trainable_variables
—layers
“metrics
 ”layer_regularization_losses
‘layer_metrics
√	variables
ƒtrainable_variables
≈regularization_losses
«__call__
+»&call_and_return_all_conditional_losses
'»"call_and_return_conditional_losses"
_generic_user_object
к
’trace_02Ћ
.__inference_embedding_84_layer_call_fn_7097260Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z’trace_0
Е
÷trace_02ж
I__inference_embedding_84_layer_call_and_return_conditional_losses_7097268Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z÷trace_0
):'2embedding_84/embeddings
(
–0"
trackable_list_wrapper
(
–0"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
„non_trainable_variables
Ўlayers
ўmetrics
 Џlayer_regularization_losses
џlayer_metrics
 	variables
Ћtrainable_variables
ћregularization_losses
ќ__call__
+ѕ&call_and_return_all_conditional_losses
'ѕ"call_and_return_conditional_losses"
_generic_user_object
к
№trace_02Ћ
.__inference_embedding_85_layer_call_fn_7097275Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z№trace_0
Е
Ёtrace_02ж
I__inference_embedding_85_layer_call_and_return_conditional_losses_7097283Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЁtrace_0
):'2embedding_85/embeddings
(
„0"
trackable_list_wrapper
(
„0"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
ёnon_trainable_variables
яlayers
аmetrics
 бlayer_regularization_losses
вlayer_metrics
—	variables
“trainable_variables
”regularization_losses
’__call__
+÷&call_and_return_all_conditional_losses
'÷"call_and_return_conditional_losses"
_generic_user_object
к
гtrace_02Ћ
.__inference_embedding_86_layer_call_fn_7097290Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zгtrace_0
Е
дtrace_02ж
I__inference_embedding_86_layer_call_and_return_conditional_losses_7097298Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zдtrace_0
):'2embedding_86/embeddings
(
ё0"
trackable_list_wrapper
(
ё0"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
еnon_trainable_variables
жlayers
зmetrics
 иlayer_regularization_losses
йlayer_metrics
Ў	variables
ўtrainable_variables
Џregularization_losses
№__call__
+Ё&call_and_return_all_conditional_losses
'Ё"call_and_return_conditional_losses"
_generic_user_object
к
кtrace_02Ћ
.__inference_embedding_89_layer_call_fn_7097305Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zкtrace_0
Е
лtrace_02ж
I__inference_embedding_89_layer_call_and_return_conditional_losses_7097313Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zлtrace_0
):'2embedding_89/embeddings
(
е0"
trackable_list_wrapper
(
е0"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
мnon_trainable_variables
нlayers
оmetrics
 пlayer_regularization_losses
рlayer_metrics
я	variables
аtrainable_variables
бregularization_losses
г__call__
+д&call_and_return_all_conditional_losses
'д"call_and_return_conditional_losses"
_generic_user_object
к
сtrace_02Ћ
.__inference_embedding_90_layer_call_fn_7097320Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zсtrace_0
Е
тtrace_02ж
I__inference_embedding_90_layer_call_and_return_conditional_losses_7097328Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zтtrace_0
):'W	2embedding_90/embeddings
"
_generic_user_object
 "
trackable_list_wrapper
j
у_initializer
ф_create_resource
х_initialize
ц_destroy_resourceR jtf.StaticHashTable
"
_generic_user_object
 "
trackable_list_wrapper
j
ч_initializer
ш_create_resource
щ_initialize
ъ_destroy_resourceR jtf.StaticHashTable
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
ыnon_trainable_variables
ьlayers
эmetrics
 юlayer_regularization_losses
€layer_metrics
м	variables
нtrainable_variables
оregularization_losses
р__call__
+с&call_and_return_all_conditional_losses
'с"call_and_return_conditional_losses"
_generic_user_object
и
Аtrace_02…
,__inference_flatten_76_layer_call_fn_7097333Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zАtrace_0
Г
Бtrace_02д
G__inference_flatten_76_layer_call_and_return_conditional_losses_7097339Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zБtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Вnon_trainable_variables
Гlayers
Дmetrics
 Еlayer_regularization_losses
Жlayer_metrics
т	variables
уtrainable_variables
фregularization_losses
ц__call__
+ч&call_and_return_all_conditional_losses
'ч"call_and_return_conditional_losses"
_generic_user_object
и
Зtrace_02…
,__inference_flatten_77_layer_call_fn_7097344Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЗtrace_0
Г
Иtrace_02д
G__inference_flatten_77_layer_call_and_return_conditional_losses_7097350Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zИtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Йnon_trainable_variables
Кlayers
Лmetrics
 Мlayer_regularization_losses
Нlayer_metrics
ш	variables
щtrainable_variables
ъregularization_losses
ь__call__
+э&call_and_return_all_conditional_losses
'э"call_and_return_conditional_losses"
_generic_user_object
и
Оtrace_02…
,__inference_flatten_78_layer_call_fn_7097355Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zОtrace_0
Г
Пtrace_02д
G__inference_flatten_78_layer_call_and_return_conditional_losses_7097361Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zПtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Рnon_trainable_variables
Сlayers
Тmetrics
 Уlayer_regularization_losses
Фlayer_metrics
ю	variables
€trainable_variables
Аregularization_losses
В__call__
+Г&call_and_return_all_conditional_losses
'Г"call_and_return_conditional_losses"
_generic_user_object
и
Хtrace_02…
,__inference_flatten_79_layer_call_fn_7097366Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zХtrace_0
Г
Цtrace_02д
G__inference_flatten_79_layer_call_and_return_conditional_losses_7097372Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЦtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Чnon_trainable_variables
Шlayers
Щmetrics
 Ъlayer_regularization_losses
Ыlayer_metrics
Д	variables
Еtrainable_variables
Жregularization_losses
И__call__
+Й&call_and_return_all_conditional_losses
'Й"call_and_return_conditional_losses"
_generic_user_object
и
Ьtrace_02…
,__inference_flatten_80_layer_call_fn_7097377Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЬtrace_0
Г
Эtrace_02д
G__inference_flatten_80_layer_call_and_return_conditional_losses_7097383Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЭtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Юnon_trainable_variables
Яlayers
†metrics
 °layer_regularization_losses
Ґlayer_metrics
К	variables
Лtrainable_variables
Мregularization_losses
О__call__
+П&call_and_return_all_conditional_losses
'П"call_and_return_conditional_losses"
_generic_user_object
и
£trace_02…
,__inference_flatten_81_layer_call_fn_7097388Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z£trace_0
Г
§trace_02д
G__inference_flatten_81_layer_call_and_return_conditional_losses_7097394Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z§trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
•non_trainable_variables
¶layers
Іmetrics
 ®layer_regularization_losses
©layer_metrics
Р	variables
Сtrainable_variables
Тregularization_losses
Ф__call__
+Х&call_and_return_all_conditional_losses
'Х"call_and_return_conditional_losses"
_generic_user_object
и
™trace_02…
,__inference_flatten_82_layer_call_fn_7097399Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z™trace_0
Г
Ђtrace_02д
G__inference_flatten_82_layer_call_and_return_conditional_losses_7097405Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЂtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
ђnon_trainable_variables
≠layers
Ѓmetrics
 ѓlayer_regularization_losses
∞layer_metrics
Ц	variables
Чtrainable_variables
Шregularization_losses
Ъ__call__
+Ы&call_and_return_all_conditional_losses
'Ы"call_and_return_conditional_losses"
_generic_user_object
и
±trace_02…
,__inference_flatten_83_layer_call_fn_7097410Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z±trace_0
Г
≤trace_02д
G__inference_flatten_83_layer_call_and_return_conditional_losses_7097416Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z≤trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
≥non_trainable_variables
іlayers
µmetrics
 ґlayer_regularization_losses
Јlayer_metrics
Ь	variables
Эtrainable_variables
Юregularization_losses
†__call__
+°&call_and_return_all_conditional_losses
'°"call_and_return_conditional_losses"
_generic_user_object
и
Єtrace_02…
,__inference_flatten_84_layer_call_fn_7097421Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЄtrace_0
Г
єtrace_02д
G__inference_flatten_84_layer_call_and_return_conditional_losses_7097427Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zєtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Їnon_trainable_variables
їlayers
Љmetrics
 љlayer_regularization_losses
Њlayer_metrics
Ґ	variables
£trainable_variables
§regularization_losses
¶__call__
+І&call_and_return_all_conditional_losses
'І"call_and_return_conditional_losses"
_generic_user_object
и
њtrace_02…
,__inference_flatten_85_layer_call_fn_7097432Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zњtrace_0
Г
јtrace_02д
G__inference_flatten_85_layer_call_and_return_conditional_losses_7097438Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zјtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Ѕnon_trainable_variables
¬layers
√metrics
 ƒlayer_regularization_losses
≈layer_metrics
®	variables
©trainable_variables
™regularization_losses
ђ__call__
+≠&call_and_return_all_conditional_losses
'≠"call_and_return_conditional_losses"
_generic_user_object
и
∆trace_02…
,__inference_flatten_86_layer_call_fn_7097443Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z∆trace_0
Г
«trace_02д
G__inference_flatten_86_layer_call_and_return_conditional_losses_7097449Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z«trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
»non_trainable_variables
…layers
 metrics
 Ћlayer_regularization_losses
ћlayer_metrics
Ѓ	variables
ѓtrainable_variables
∞regularization_losses
≤__call__
+≥&call_and_return_all_conditional_losses
'≥"call_and_return_conditional_losses"
_generic_user_object
и
Ќtrace_02…
,__inference_flatten_89_layer_call_fn_7097454Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЌtrace_0
Г
ќtrace_02д
G__inference_flatten_89_layer_call_and_return_conditional_losses_7097460Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zќtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
ѕnon_trainable_variables
–layers
—metrics
 “layer_regularization_losses
”layer_metrics
і	variables
µtrainable_variables
ґregularization_losses
Є__call__
+є&call_and_return_all_conditional_losses
'є"call_and_return_conditional_losses"
_generic_user_object
и
‘trace_02…
,__inference_flatten_90_layer_call_fn_7097465Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z‘trace_0
Г
’trace_02д
G__inference_flatten_90_layer_call_and_return_conditional_losses_7097471Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z’trace_0
"
_generic_user_object
 "
trackable_list_wrapper
j
÷_initializer
„_create_resource
Ў_initialize
ў_destroy_resourceR jtf.StaticHashTable
(
√0"
trackable_list_wrapper
(
√0"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Џnon_trainable_variables
џlayers
№metrics
 Ёlayer_regularization_losses
ёlayer_metrics
љ	variables
Њtrainable_variables
њregularization_losses
Ѕ__call__
+¬&call_and_return_all_conditional_losses
'¬"call_and_return_conditional_losses"
_generic_user_object
к
яtrace_02Ћ
.__inference_embedding_87_layer_call_fn_7097478Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zяtrace_0
Е
аtrace_02ж
I__inference_embedding_87_layer_call_and_return_conditional_losses_7097486Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zаtrace_0
):'2embedding_87/embeddings
(
 0"
trackable_list_wrapper
(
 0"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
бnon_trainable_variables
вlayers
гmetrics
 дlayer_regularization_losses
еlayer_metrics
ƒ	variables
≈trainable_variables
∆regularization_losses
»__call__
+…&call_and_return_all_conditional_losses
'…"call_and_return_conditional_losses"
_generic_user_object
к
жtrace_02Ћ
.__inference_embedding_88_layer_call_fn_7097493Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zжtrace_0
Е
зtrace_02ж
I__inference_embedding_88_layer_call_and_return_conditional_losses_7097501Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zзtrace_0
):'2embedding_88/embeddings
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
иnon_trainable_variables
йlayers
кmetrics
 лlayer_regularization_losses
мlayer_metrics
Ћ	variables
ћtrainable_variables
Ќregularization_losses
ѕ__call__
+–&call_and_return_all_conditional_losses
'–"call_and_return_conditional_losses"
_generic_user_object
м
нtrace_02Ќ
0__inference_concatenate_31_layer_call_fn_7097520Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zнtrace_0
З
оtrace_02и
K__inference_concatenate_31_layer_call_and_return_conditional_losses_7097540Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zоtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
пnon_trainable_variables
рlayers
сmetrics
 тlayer_regularization_losses
уlayer_metrics
—	variables
“trainable_variables
”regularization_losses
’__call__
+÷&call_and_return_all_conditional_losses
'÷"call_and_return_conditional_losses"
_generic_user_object
м
фtrace_02Ќ
0__inference_concatenate_33_layer_call_fn_7097554Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zфtrace_0
З
хtrace_02и
K__inference_concatenate_33_layer_call_and_return_conditional_losses_7097569Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zхtrace_0
"
_generic_user_object
 "
trackable_list_wrapper
j
ц_initializer
ч_create_resource
ш_initialize
щ_destroy_resourceR jtf.StaticHashTable
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
ъnon_trainable_variables
ыlayers
ьmetrics
 эlayer_regularization_losses
юlayer_metrics
Џ	variables
џtrainable_variables
№regularization_losses
ё__call__
+я&call_and_return_all_conditional_losses
'я"call_and_return_conditional_losses"
_generic_user_object
и
€trace_02…
,__inference_flatten_87_layer_call_fn_7097574Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z€trace_0
Г
Аtrace_02д
G__inference_flatten_87_layer_call_and_return_conditional_losses_7097580Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zАtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Бnon_trainable_variables
Вlayers
Гmetrics
 Дlayer_regularization_losses
Еlayer_metrics
а	variables
бtrainable_variables
вregularization_losses
д__call__
+е&call_and_return_all_conditional_losses
'е"call_and_return_conditional_losses"
_generic_user_object
и
Жtrace_02…
,__inference_flatten_88_layer_call_fn_7097585Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЖtrace_0
Г
Зtrace_02д
G__inference_flatten_88_layer_call_and_return_conditional_losses_7097591Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЗtrace_0
0
м0
н1"
trackable_list_wrapper
0
м0
н1"
trackable_list_wrapper
(
ї0"
trackable_list_wrapper
Є
Иnon_trainable_variables
Йlayers
Кmetrics
 Лlayer_regularization_losses
Мlayer_metrics
ж	variables
зtrainable_variables
иregularization_losses
к__call__
+л&call_and_return_all_conditional_losses
'л"call_and_return_conditional_losses"
_generic_user_object
м
Нtrace_02Ќ
0__inference_user_embedding_layer_call_fn_7097600Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zНtrace_0
З
Оtrace_02и
K__inference_user_embedding_layer_call_and_return_conditional_losses_7097614Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zОtrace_0
(:&	ђ2user_embedding/kernel
": ђ2user_embedding/bias
0
ф0
х1"
trackable_list_wrapper
0
ф0
х1"
trackable_list_wrapper
(
Љ0"
trackable_list_wrapper
Є
Пnon_trainable_variables
Рlayers
Сmetrics
 Тlayer_regularization_losses
Уlayer_metrics
о	variables
пtrainable_variables
рregularization_losses
т__call__
+у&call_and_return_all_conditional_losses
'у"call_and_return_conditional_losses"
_generic_user_object
м
Фtrace_02Ќ
0__inference_food_embedding_layer_call_fn_7097623Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zФtrace_0
З
Хtrace_02и
K__inference_food_embedding_layer_call_and_return_conditional_losses_7097637Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zХtrace_0
):'
£ђ2food_embedding/kernel
": ђ2food_embedding/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Цnon_trainable_variables
Чlayers
Шmetrics
 Щlayer_regularization_losses
Ъlayer_metrics
ц	variables
чtrainable_variables
шregularization_losses
ъ__call__
+ы&call_and_return_all_conditional_losses
'ы"call_and_return_conditional_losses"
_generic_user_object
м
Ыtrace_02Ќ
0__inference_concatenate_32_layer_call_fn_7097646Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЫtrace_0
З
Ьtrace_02и
K__inference_concatenate_32_layer_call_and_return_conditional_losses_7097656Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЬtrace_0
0
В0
Г1"
trackable_list_wrapper
0
В0
Г1"
trackable_list_wrapper
(
љ0"
trackable_list_wrapper
Є
Эnon_trainable_variables
Юlayers
Яmetrics
 †layer_regularization_losses
°layer_metrics
ь	variables
эtrainable_variables
юregularization_losses
А__call__
+Б&call_and_return_all_conditional_losses
'Б"call_and_return_conditional_losses"
_generic_user_object
п
Ґtrace_02–
3__inference_context_embedding_layer_call_fn_7097665Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zҐtrace_0
К
£trace_02л
N__inference_context_embedding_layer_call_and_return_conditional_losses_7097679Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z£trace_0
+:)	Ш2context_embedding/kernel
$:"2context_embedding/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
§non_trainable_variables
•layers
¶metrics
 Іlayer_regularization_losses
®layer_metrics
Д	variables
Еtrainable_variables
Жregularization_losses
И__call__
+Й&call_and_return_all_conditional_losses
'Й"call_and_return_conditional_losses"
_generic_user_object
д
©trace_02≈
(__inference_dot_12_layer_call_fn_7097685Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z©trace_0
€
™trace_02а
C__inference_dot_12_layer_call_and_return_conditional_losses_7097711Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z™trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Є
Ђnon_trainable_variables
ђlayers
≠metrics
 Ѓlayer_regularization_losses
ѓlayer_metrics
К	variables
Лtrainable_variables
Мregularization_losses
О__call__
+П&call_and_return_all_conditional_losses
'П"call_and_return_conditional_losses"
_generic_user_object
м
∞trace_02Ќ
0__inference_concatenate_34_layer_call_fn_7097719Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z∞trace_0
З
±trace_02и
K__inference_concatenate_34_layer_call_and_return_conditional_losses_7097728Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z±trace_0
@
Ч0
Ш1
Щ2
Ъ3"
trackable_list_wrapper
0
Ч0
Ш1"
trackable_list_wrapper
 "
trackable_list_wrapper
Є
≤non_trainable_variables
≥layers
іmetrics
 µlayer_regularization_losses
ґlayer_metrics
Р	variables
Сtrainable_variables
Тregularization_losses
Ф__call__
+Х&call_and_return_all_conditional_losses
'Х"call_and_return_conditional_losses"
_generic_user_object
з
Јtrace_0
Єtrace_12ђ
8__inference_batch_normalization_12_layer_call_fn_7097741
8__inference_batch_normalization_12_layer_call_fn_7097754µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЈtrace_0zЄtrace_1
Э
єtrace_0
Їtrace_12в
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_7097788
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_7097808µ
Ѓ≤™
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsҐ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zєtrace_0zЇtrace_1
 "
trackable_list_wrapper
+:)й2batch_normalization_12/gamma
*:(й2batch_normalization_12/beta
3:1й (2"batch_normalization_12/moving_mean
7:5й (2&batch_normalization_12/moving_variance
0
°0
Ґ1"
trackable_list_wrapper
0
°0
Ґ1"
trackable_list_wrapper
(
Њ0"
trackable_list_wrapper
Є
їnon_trainable_variables
Љlayers
љmetrics
 Њlayer_regularization_losses
њlayer_metrics
Ы	variables
Ьtrainable_variables
Эregularization_losses
Я__call__
+†&call_and_return_all_conditional_losses
'†"call_and_return_conditional_losses"
_generic_user_object
и
јtrace_02…
,__inference_fc_layer_0_layer_call_fn_7097817Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zјtrace_0
Г
Ѕtrace_02д
G__inference_fc_layer_0_layer_call_and_return_conditional_losses_7097832Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЅtrace_0
%:#
йА2fc_layer_0/kernel
:А2fc_layer_0/bias
0
©0
™1"
trackable_list_wrapper
0
©0
™1"
trackable_list_wrapper
(
њ0"
trackable_list_wrapper
Є
¬non_trainable_variables
√layers
ƒmetrics
 ≈layer_regularization_losses
∆layer_metrics
£	variables
§trainable_variables
•regularization_losses
І__call__
+®&call_and_return_all_conditional_losses
'®"call_and_return_conditional_losses"
_generic_user_object
и
«trace_02…
,__inference_fc_layer_1_layer_call_fn_7097841Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z«trace_0
Г
»trace_02д
G__inference_fc_layer_1_layer_call_and_return_conditional_losses_7097856Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z»trace_0
$:"	А@2fc_layer_1/kernel
:@2fc_layer_1/bias
0
±0
≤1"
trackable_list_wrapper
0
±0
≤1"
trackable_list_wrapper
(
ј0"
trackable_list_wrapper
Є
…non_trainable_variables
 layers
Ћmetrics
 ћlayer_regularization_losses
Ќlayer_metrics
Ђ	variables
ђtrainable_variables
≠regularization_losses
ѓ__call__
+∞&call_and_return_all_conditional_losses
'∞"call_and_return_conditional_losses"
_generic_user_object
и
ќtrace_02…
,__inference_fc_layer_2_layer_call_fn_7097865Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zќtrace_0
Г
ѕtrace_02д
G__inference_fc_layer_2_layer_call_and_return_conditional_losses_7097880Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zѕtrace_0
#:!@ 2fc_layer_2/kernel
: 2fc_layer_2/bias
0
є0
Ї1"
trackable_list_wrapper
0
є0
Ї1"
trackable_list_wrapper
(
Ѕ0"
trackable_list_wrapper
Є
–non_trainable_variables
—layers
“metrics
 ”layer_regularization_losses
‘layer_metrics
≥	variables
іtrainable_variables
µregularization_losses
Ј__call__
+Є&call_and_return_all_conditional_losses
'Є"call_and_return_conditional_losses"
_generic_user_object
ж
’trace_02«
*__inference_output_0_layer_call_fn_7097889Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z’trace_0
Б
÷trace_02в
E__inference_output_0_layer_call_and_return_conditional_losses_7097904Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z÷trace_0
!: 2output_0/kernel
:2output_0/bias
–
„trace_02±
__inference_loss_fn_0_7097912П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ z„trace_0
–
Ўtrace_02±
__inference_loss_fn_1_7097920П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zЎtrace_0
–
ўtrace_02±
__inference_loss_fn_2_7097928П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zўtrace_0
–
Џtrace_02±
__inference_loss_fn_3_7097936П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zЏtrace_0
–
џtrace_02±
__inference_loss_fn_4_7097944П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zџtrace_0
–
№trace_02±
__inference_loss_fn_5_7097952П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ z№trace_0
–
Ёtrace_02±
__inference_loss_fn_6_7097960П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zЁtrace_0
0
Щ0
Ъ1"
trackable_list_wrapper
ж
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37
'38
(39
)40
*41
+42
,43
-44
.45
/46
047
148
249
350
451
552
653
754
855
956
:57
;58
<59
=60
>61
?62
@63
A64
B65
C66
D67
E68
F69
G70
H71
I72
J73
K74
L75
M76
N77
O78
P79
Q80
R81
S82
T83
U84
V85
W86
X87
Y88
Z89"
trackable_list_wrapper
@
ё0
я1
а2
б3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
Ћ	
Ћ	capture_1
ћ	capture_3
Ќ	capture_5
ќ	capture_7
ѕ	capture_9
–
capture_11
—
capture_13
“
capture_15
”
capture_17
‘
capture_19
’
capture_21
÷
capture_23
„
capture_25
Ў
capture_27
ў
capture_29
Џ
capture_46
џ
capture_48BР
*__inference_model_12_layer_call_fn_7096254BMI	age_range	allergensallergycaloriescarbohydratesclinical_gendercultural_factorcultural_restrictioncurrent_daily_caloriescurrent_working_status
day_number
embeddings	ethnicityfatfiberheight
life_stylemarital_statusmeal_type_ynext_BMInutrition_goalplace_of_meal_consumptionpriceprojected_daily_caloriesprotein$social_situation_of_meal_consumptiontastetime_of_meal_consumptionweight"ђ
•≤°
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЋ	capture_1zћ	capture_3zЌ	capture_5zќ	capture_7zѕ	capture_9z–
capture_11z—
capture_13z“
capture_15z”
capture_17z‘
capture_19z’
capture_21z÷
capture_23z„
capture_25zЎ
capture_27zў
capture_29zЏ
capture_46zџ
capture_48
Ћ	
Ћ	capture_1
ћ	capture_3
Ќ	capture_5
ќ	capture_7
ѕ	capture_9
–
capture_11
—
capture_13
“
capture_15
”
capture_17
‘
capture_19
’
capture_21
÷
capture_23
„
capture_25
Ў
capture_27
ў
capture_29
Џ
capture_46
џ
capture_48BР
*__inference_model_12_layer_call_fn_7096422BMI	age_range	allergensallergycaloriescarbohydratesclinical_gendercultural_factorcultural_restrictioncurrent_daily_caloriescurrent_working_status
day_number
embeddings	ethnicityfatfiberheight
life_stylemarital_statusmeal_type_ynext_BMInutrition_goalplace_of_meal_consumptionpriceprojected_daily_caloriesprotein$social_situation_of_meal_consumptiontastetime_of_meal_consumptionweight"ђ
•≤°
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЋ	capture_1zћ	capture_3zЌ	capture_5zќ	capture_7zѕ	capture_9z–
capture_11z—
capture_13z“
capture_15z”
capture_17z‘
capture_19z’
capture_21z÷
capture_23z„
capture_25zЎ
capture_27zў
capture_29zЏ
capture_46zџ
capture_48
ж	
Ћ	capture_1
ћ	capture_3
Ќ	capture_5
ќ	capture_7
ѕ	capture_9
–
capture_11
—
capture_13
“
capture_15
”
capture_17
‘
capture_19
’
capture_21
÷
capture_23
„
capture_25
Ў
capture_27
ў
capture_29
Џ
capture_46
џ
capture_48BЂ
E__inference_model_12_layer_call_and_return_conditional_losses_7095682BMI	age_range	allergensallergycaloriescarbohydratesclinical_gendercultural_factorcultural_restrictioncurrent_daily_caloriescurrent_working_status
day_number
embeddings	ethnicityfatfiberheight
life_stylemarital_statusmeal_type_ynext_BMInutrition_goalplace_of_meal_consumptionpriceprojected_daily_caloriesprotein$social_situation_of_meal_consumptiontastetime_of_meal_consumptionweight"ђ
•≤°
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЋ	capture_1zћ	capture_3zЌ	capture_5zќ	capture_7zѕ	capture_9z–
capture_11z—
capture_13z“
capture_15z”
capture_17z‘
capture_19z’
capture_21z÷
capture_23z„
capture_25zЎ
capture_27zў
capture_29zЏ
capture_46zџ
capture_48
ж	
Ћ	capture_1
ћ	capture_3
Ќ	capture_5
ќ	capture_7
ѕ	capture_9
–
capture_11
—
capture_13
“
capture_15
”
capture_17
‘
capture_19
’
capture_21
÷
capture_23
„
capture_25
Ў
capture_27
ў
capture_29
Џ
capture_46
џ
capture_48BЂ
E__inference_model_12_layer_call_and_return_conditional_losses_7096086BMI	age_range	allergensallergycaloriescarbohydratesclinical_gendercultural_factorcultural_restrictioncurrent_daily_caloriescurrent_working_status
day_number
embeddings	ethnicityfatfiberheight
life_stylemarital_statusmeal_type_ynext_BMInutrition_goalplace_of_meal_consumptionpriceprojected_daily_caloriesprotein$social_situation_of_meal_consumptiontastetime_of_meal_consumptionweight"ђ
•≤°
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЋ	capture_1zћ	capture_3zЌ	capture_5zќ	capture_7zѕ	capture_9z–
capture_11z—
capture_13z“
capture_15z”
capture_17z‘
capture_19z’
capture_21z÷
capture_23z„
capture_25zЎ
capture_27zў
capture_29zЏ
capture_46zџ
capture_48
"J

Const_50jtf.TrackableConstant
"J

Const_49jtf.TrackableConstant
"J

Const_48jtf.TrackableConstant
"J

Const_47jtf.TrackableConstant
"J

Const_46jtf.TrackableConstant
"J

Const_45jtf.TrackableConstant
"J

Const_44jtf.TrackableConstant
"J

Const_43jtf.TrackableConstant
"J

Const_42jtf.TrackableConstant
"J

Const_41jtf.TrackableConstant
"J

Const_40jtf.TrackableConstant
"J

Const_39jtf.TrackableConstant
"J

Const_38jtf.TrackableConstant
"J

Const_37jtf.TrackableConstant
"J

Const_36jtf.TrackableConstant
"J

Const_35jtf.TrackableConstant
"J

Const_34jtf.TrackableConstant
Ќ
Ё0
в1
г2
д3
е4
ж5
з6
и7
й8
к9
л10
м11
н12
о13
п14
р15
с16
т17
у18
ф19
х20
ц21
ч22
ш23
щ24
ъ25
ы26
ь27
э28
ю29
€30
А31
Б32
В33
Г34
Д35
Е36
Ж37
З38
И39
Й40
К41
Л42
М43
Н44
О45
П46
Р47
С48
Т49
У50
Ф51
Х52
Ц53
Ч54
Ш55
Щ56
Ъ57
Ы58
Ь59
Э60
Ю61
Я62"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
≠
в0
д1
ж2
и3
к4
м5
о6
р7
т8
ф9
ц10
ш11
ъ12
ь13
ю14
А15
В16
Д17
Ж18
И19
К20
М21
О22
Р23
Т24
Ф25
Ц26
Ш27
Ъ28
Ь29
Ю30"
trackable_list_wrapper
≠
г0
е1
з2
й3
л4
н5
п6
с7
у8
х9
ч10
щ11
ы12
э13
€14
Б15
Г16
Е17
З18
Й19
Л20
Н21
П22
С23
У24
Х25
Ч26
Щ27
Ы28
Э29
Я30"
trackable_list_wrapper
Ё
†trace_0
°trace_1
Ґtrace_2
£trace_3
§trace_4
•trace_5
¶trace_6
Іtrace_7
®trace_8
©trace_9
™trace_10
Ђtrace_11
ђtrace_12
≠trace_13
Ѓtrace_14
ѓtrace_15
∞trace_16
±trace_17
≤trace_18
≥trace_19
іtrace_20
µtrace_21
ґtrace_22
Јtrace_23
Єtrace_24
єtrace_25
Їtrace_26
їtrace_27
Љtrace_28
љtrace_29
Њtrace_302ћ

$__inference__update_step_xla_7096955
$__inference__update_step_xla_7096962
$__inference__update_step_xla_7096969
$__inference__update_step_xla_7096976
$__inference__update_step_xla_7096983
$__inference__update_step_xla_7096990
$__inference__update_step_xla_7096997
$__inference__update_step_xla_7097004
$__inference__update_step_xla_7097011
$__inference__update_step_xla_7097018
$__inference__update_step_xla_7097025
$__inference__update_step_xla_7097032
$__inference__update_step_xla_7097039
$__inference__update_step_xla_7097046
$__inference__update_step_xla_7097053
$__inference__update_step_xla_7097058
$__inference__update_step_xla_7097063
$__inference__update_step_xla_7097068
$__inference__update_step_xla_7097073
$__inference__update_step_xla_7097078
$__inference__update_step_xla_7097083
$__inference__update_step_xla_7097088
$__inference__update_step_xla_7097093
$__inference__update_step_xla_7097098
$__inference__update_step_xla_7097103
$__inference__update_step_xla_7097108
$__inference__update_step_xla_7097113
$__inference__update_step_xla_7097118
$__inference__update_step_xla_7097123
$__inference__update_step_xla_7097128
$__inference__update_step_xla_7097133ѓ
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 0z†trace_0z°trace_1zҐtrace_2z£trace_3z§trace_4z•trace_5z¶trace_6zІtrace_7z®trace_8z©trace_9z™trace_10zЂtrace_11zђtrace_12z≠trace_13zЃtrace_14zѓtrace_15z∞trace_16z±trace_17z≤trace_18z≥trace_19zіtrace_20zµtrace_21zґtrace_22zЈtrace_23zЄtrace_24zєtrace_25zЇtrace_26zїtrace_27zЉtrace_28zљtrace_29zЊtrace_30
Ш
Ћ	capture_1
ћ	capture_3
Ќ	capture_5
ќ	capture_7
ѕ	capture_9
–
capture_11
—
capture_13
“
capture_15
”
capture_17
‘
capture_19
’
capture_21
÷
capture_23
„
capture_25
Ў
capture_27
ў
capture_29
Џ
capture_46
џ
capture_48BЁ
%__inference_signature_wrapper_7096920BMI	age_range	allergensallergycaloriescarbohydratesclinical_gendercultural_factorcultural_restrictioncurrent_daily_caloriescurrent_working_status
day_number
embeddings	ethnicityfatfiberheight
life_stylemarital_statusmeal_type_ynext_BMInutrition_goalplace_of_meal_consumptionpriceprojected_daily_caloriesprotein$social_situation_of_meal_consumptiontastetime_of_meal_consumptionweight"А
щ≤х
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 В

kwonlyargsуЪп
jBMI
j	age_range
j	allergens
	jallergy

jcalories
jcarbohydrates
jclinical_gender
jcultural_factor
jcultural_restriction
jcurrent_daily_calories
jcurrent_working_status
j
day_number
j
embeddings
j	ethnicity
jfat
jfiber
jheight
j
life_style
jmarital_status
jmeal_type_y

jnext_BMI
jnutrition_goal
jplace_of_meal_consumption
jprice
jprojected_daily_calories
	jprotein
&j$social_situation_of_meal_consumption
jtaste
jtime_of_meal_consumption
jweight
kwonlydefaults
 
annotations™ *
 zЋ	capture_1zћ	capture_3zЌ	capture_5zќ	capture_7zѕ	capture_9z–
capture_11z—
capture_13z“
capture_15z”
capture_17z‘
capture_19z’
capture_21z÷
capture_23z„
capture_25zЎ
capture_27zў
capture_29zЏ
capture_46zџ
capture_48
"
_generic_user_object
ѕ
њtrace_02∞
__inference__creator_7097964П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zњtrace_0
”
јtrace_02і
 __inference__initializer_7097971П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zјtrace_0
—
Ѕtrace_02≤
__inference__destroyer_7097975П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zЅtrace_0
"
_generic_user_object
ѕ
¬trace_02∞
__inference__creator_7097979П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ z¬trace_0
”
√trace_02і
 __inference__initializer_7097986П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ z√trace_0
—
ƒtrace_02≤
__inference__destroyer_7097990П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zƒtrace_0
"
_generic_user_object
ѕ
≈trace_02∞
__inference__creator_7097994П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ z≈trace_0
”
∆trace_02і
 __inference__initializer_7098001П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ z∆trace_0
—
«trace_02≤
__inference__destroyer_7098005П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ z«trace_0
"
_generic_user_object
ѕ
»trace_02∞
__inference__creator_7098009П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ z»trace_0
”
…trace_02і
 __inference__initializer_7098016П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ z…trace_0
—
 trace_02≤
__inference__destroyer_7098020П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ z trace_0
"
_generic_user_object
ѕ
Ћtrace_02∞
__inference__creator_7098024П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zЋtrace_0
”
ћtrace_02і
 __inference__initializer_7098031П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zћtrace_0
—
Ќtrace_02≤
__inference__destroyer_7098035П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zЌtrace_0
"
_generic_user_object
ѕ
ќtrace_02∞
__inference__creator_7098039П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zќtrace_0
”
ѕtrace_02і
 __inference__initializer_7098046П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zѕtrace_0
—
–trace_02≤
__inference__destroyer_7098050П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ z–trace_0
"
_generic_user_object
ѕ
—trace_02∞
__inference__creator_7098054П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ z—trace_0
”
“trace_02і
 __inference__initializer_7098061П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ z“trace_0
—
”trace_02≤
__inference__destroyer_7098065П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ z”trace_0
"
_generic_user_object
ѕ
‘trace_02∞
__inference__creator_7098069П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ z‘trace_0
”
’trace_02і
 __inference__initializer_7098076П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ z’trace_0
—
÷trace_02≤
__inference__destroyer_7098080П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ z÷trace_0
"
_generic_user_object
ѕ
„trace_02∞
__inference__creator_7098084П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ z„trace_0
”
Ўtrace_02і
 __inference__initializer_7098091П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zЎtrace_0
—
ўtrace_02≤
__inference__destroyer_7098095П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zўtrace_0
"
_generic_user_object
ѕ
Џtrace_02∞
__inference__creator_7098099П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zЏtrace_0
”
џtrace_02і
 __inference__initializer_7098106П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zџtrace_0
—
№trace_02≤
__inference__destroyer_7098110П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ z№trace_0
"
_generic_user_object
ѕ
Ёtrace_02∞
__inference__creator_7098114П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zЁtrace_0
”
ёtrace_02і
 __inference__initializer_7098121П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zёtrace_0
—
яtrace_02≤
__inference__destroyer_7098125П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zяtrace_0
"
_generic_user_object
ѕ
аtrace_02∞
__inference__creator_7098129П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zаtrace_0
”
бtrace_02і
 __inference__initializer_7098136П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zбtrace_0
—
вtrace_02≤
__inference__destroyer_7098140П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zвtrace_0
"
_generic_user_object
ѕ
гtrace_02∞
__inference__creator_7098144П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zгtrace_0
”
дtrace_02і
 __inference__initializer_7098151П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zдtrace_0
—
еtrace_02≤
__inference__destroyer_7098155П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zеtrace_0
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
ЎB’
.__inference_embedding_76_layer_call_fn_7097140inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
уBр
I__inference_embedding_76_layer_call_and_return_conditional_losses_7097148inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
ЎB’
.__inference_embedding_77_layer_call_fn_7097155inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
уBр
I__inference_embedding_77_layer_call_and_return_conditional_losses_7097163inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
ЎB’
.__inference_embedding_78_layer_call_fn_7097170inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
уBр
I__inference_embedding_78_layer_call_and_return_conditional_losses_7097178inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
ЎB’
.__inference_embedding_79_layer_call_fn_7097185inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
уBр
I__inference_embedding_79_layer_call_and_return_conditional_losses_7097193inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
ЎB’
.__inference_embedding_80_layer_call_fn_7097200inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
уBр
I__inference_embedding_80_layer_call_and_return_conditional_losses_7097208inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
ЎB’
.__inference_embedding_81_layer_call_fn_7097215inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
уBр
I__inference_embedding_81_layer_call_and_return_conditional_losses_7097223inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
ЎB’
.__inference_embedding_82_layer_call_fn_7097230inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
уBр
I__inference_embedding_82_layer_call_and_return_conditional_losses_7097238inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
ЎB’
.__inference_embedding_83_layer_call_fn_7097245inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
уBр
I__inference_embedding_83_layer_call_and_return_conditional_losses_7097253inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
ЎB’
.__inference_embedding_84_layer_call_fn_7097260inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
уBр
I__inference_embedding_84_layer_call_and_return_conditional_losses_7097268inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
ЎB’
.__inference_embedding_85_layer_call_fn_7097275inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
уBр
I__inference_embedding_85_layer_call_and_return_conditional_losses_7097283inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
ЎB’
.__inference_embedding_86_layer_call_fn_7097290inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
уBр
I__inference_embedding_86_layer_call_and_return_conditional_losses_7097298inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
ЎB’
.__inference_embedding_89_layer_call_fn_7097305inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
уBр
I__inference_embedding_89_layer_call_and_return_conditional_losses_7097313inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
ЎB’
.__inference_embedding_90_layer_call_fn_7097320inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
уBр
I__inference_embedding_90_layer_call_and_return_conditional_losses_7097328inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
"
_generic_user_object
ѕ
жtrace_02∞
__inference__creator_7098159П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zжtrace_0
”
зtrace_02і
 __inference__initializer_7098166П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zзtrace_0
—
иtrace_02≤
__inference__destroyer_7098170П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zиtrace_0
"
_generic_user_object
ѕ
йtrace_02∞
__inference__creator_7098174П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zйtrace_0
”
кtrace_02і
 __inference__initializer_7098181П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zкtrace_0
—
лtrace_02≤
__inference__destroyer_7098185П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zлtrace_0
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
÷B”
,__inference_flatten_76_layer_call_fn_7097333inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
сBо
G__inference_flatten_76_layer_call_and_return_conditional_losses_7097339inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
÷B”
,__inference_flatten_77_layer_call_fn_7097344inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
сBо
G__inference_flatten_77_layer_call_and_return_conditional_losses_7097350inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
÷B”
,__inference_flatten_78_layer_call_fn_7097355inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
сBо
G__inference_flatten_78_layer_call_and_return_conditional_losses_7097361inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
÷B”
,__inference_flatten_79_layer_call_fn_7097366inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
сBо
G__inference_flatten_79_layer_call_and_return_conditional_losses_7097372inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
÷B”
,__inference_flatten_80_layer_call_fn_7097377inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
сBо
G__inference_flatten_80_layer_call_and_return_conditional_losses_7097383inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
÷B”
,__inference_flatten_81_layer_call_fn_7097388inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
сBо
G__inference_flatten_81_layer_call_and_return_conditional_losses_7097394inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
÷B”
,__inference_flatten_82_layer_call_fn_7097399inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
сBо
G__inference_flatten_82_layer_call_and_return_conditional_losses_7097405inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
÷B”
,__inference_flatten_83_layer_call_fn_7097410inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
сBо
G__inference_flatten_83_layer_call_and_return_conditional_losses_7097416inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
÷B”
,__inference_flatten_84_layer_call_fn_7097421inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
сBо
G__inference_flatten_84_layer_call_and_return_conditional_losses_7097427inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
÷B”
,__inference_flatten_85_layer_call_fn_7097432inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
сBо
G__inference_flatten_85_layer_call_and_return_conditional_losses_7097438inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
÷B”
,__inference_flatten_86_layer_call_fn_7097443inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
сBо
G__inference_flatten_86_layer_call_and_return_conditional_losses_7097449inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
÷B”
,__inference_flatten_89_layer_call_fn_7097454inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
сBо
G__inference_flatten_89_layer_call_and_return_conditional_losses_7097460inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
÷B”
,__inference_flatten_90_layer_call_fn_7097465inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
сBо
G__inference_flatten_90_layer_call_and_return_conditional_losses_7097471inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
"
_generic_user_object
ѕ
мtrace_02∞
__inference__creator_7098189П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zмtrace_0
”
нtrace_02і
 __inference__initializer_7098196П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zнtrace_0
—
оtrace_02≤
__inference__destroyer_7098200П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zоtrace_0
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
ЎB’
.__inference_embedding_87_layer_call_fn_7097478inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
уBр
I__inference_embedding_87_layer_call_and_return_conditional_losses_7097486inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
ЎB’
.__inference_embedding_88_layer_call_fn_7097493inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
уBр
I__inference_embedding_88_layer_call_and_return_conditional_losses_7097501inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
нBк
0__inference_concatenate_31_layer_call_fn_7097520inputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11	inputs_12	inputs_13	inputs_14"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ИBЕ
K__inference_concatenate_31_layer_call_and_return_conditional_losses_7097540inputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11	inputs_12	inputs_13	inputs_14"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
ґB≥
0__inference_concatenate_33_layer_call_fn_7097554inputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9
"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
—Bќ
K__inference_concatenate_33_layer_call_and_return_conditional_losses_7097569inputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9
"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
"
_generic_user_object
ѕ
пtrace_02∞
__inference__creator_7098204П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zпtrace_0
”
рtrace_02і
 __inference__initializer_7098211П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zрtrace_0
—
сtrace_02≤
__inference__destroyer_7098215П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zсtrace_0
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
÷B”
,__inference_flatten_87_layer_call_fn_7097574inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
сBо
G__inference_flatten_87_layer_call_and_return_conditional_losses_7097580inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
÷B”
,__inference_flatten_88_layer_call_fn_7097585inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
сBо
G__inference_flatten_88_layer_call_and_return_conditional_losses_7097591inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
ї0"
trackable_list_wrapper
 "
trackable_dict_wrapper
ЏB„
0__inference_user_embedding_layer_call_fn_7097600inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
хBт
K__inference_user_embedding_layer_call_and_return_conditional_losses_7097614inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
Љ0"
trackable_list_wrapper
 "
trackable_dict_wrapper
ЏB„
0__inference_food_embedding_layer_call_fn_7097623inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
хBт
K__inference_food_embedding_layer_call_and_return_conditional_losses_7097637inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
ДBБ
0__inference_concatenate_32_layer_call_fn_7097646inputs_0inputs_1inputs_2inputs_3inputs_4"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЯBЬ
K__inference_concatenate_32_layer_call_and_return_conditional_losses_7097656inputs_0inputs_1inputs_2inputs_3inputs_4"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
љ0"
trackable_list_wrapper
 "
trackable_dict_wrapper
ЁBЏ
3__inference_context_embedding_layer_call_fn_7097665inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
шBх
N__inference_context_embedding_layer_call_and_return_conditional_losses_7097679inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
ёBџ
(__inference_dot_12_layer_call_fn_7097685inputs_0inputs_1"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
щBц
C__inference_dot_12_layer_call_and_return_conditional_losses_7097711inputs_0inputs_1"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
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
ъBч
0__inference_concatenate_34_layer_call_fn_7097719inputs_0inputs_1inputs_2inputs_3"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ХBТ
K__inference_concatenate_34_layer_call_and_return_conditional_losses_7097728inputs_0inputs_1inputs_2inputs_3"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
0
Щ0
Ъ1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
цBу
8__inference_batch_normalization_12_layer_call_fn_7097741inputs"ђ
•≤°
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
цBу
8__inference_batch_normalization_12_layer_call_fn_7097754inputs"ђ
•≤°
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
СBО
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_7097788inputs"ђ
•≤°
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
СBО
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_7097808inputs"ђ
•≤°
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
Њ0"
trackable_list_wrapper
 "
trackable_dict_wrapper
÷B”
,__inference_fc_layer_0_layer_call_fn_7097817inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
сBо
G__inference_fc_layer_0_layer_call_and_return_conditional_losses_7097832inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
њ0"
trackable_list_wrapper
 "
trackable_dict_wrapper
÷B”
,__inference_fc_layer_1_layer_call_fn_7097841inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
сBо
G__inference_fc_layer_1_layer_call_and_return_conditional_losses_7097856inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
ј0"
trackable_list_wrapper
 "
trackable_dict_wrapper
÷B”
,__inference_fc_layer_2_layer_call_fn_7097865inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
сBо
G__inference_fc_layer_2_layer_call_and_return_conditional_losses_7097880inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
Ѕ0"
trackable_list_wrapper
 "
trackable_dict_wrapper
‘B—
*__inference_output_0_layer_call_fn_7097889inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
пBм
E__inference_output_0_layer_call_and_return_conditional_losses_7097904inputs"Ш
С≤Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
іB±
__inference_loss_fn_0_7097912"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
іB±
__inference_loss_fn_1_7097920"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
іB±
__inference_loss_fn_2_7097928"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
іB±
__inference_loss_fn_3_7097936"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
іB±
__inference_loss_fn_4_7097944"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
іB±
__inference_loss_fn_5_7097952"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
іB±
__inference_loss_fn_6_7097960"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
R
т	variables
у	keras_api

фtotal

хcount"
_tf_keras_metric
c
ц	variables
ч	keras_api

шtotal

щcount
ъ
_fn_kwargs"
_tf_keras_metric
v
ы	variables
ь	keras_api
э
thresholds
юtrue_positives
€false_positives"
_tf_keras_metric
v
А	variables
Б	keras_api
В
thresholds
Гtrue_positives
Дfalse_negatives"
_tf_keras_metric
.:,2Adam/m/embedding_76/embeddings
.:,2Adam/v/embedding_76/embeddings
.:,2Adam/m/embedding_77/embeddings
.:,2Adam/v/embedding_77/embeddings
.:,2Adam/m/embedding_78/embeddings
.:,2Adam/v/embedding_78/embeddings
.:,2Adam/m/embedding_79/embeddings
.:,2Adam/v/embedding_79/embeddings
.:,2Adam/m/embedding_80/embeddings
.:,2Adam/v/embedding_80/embeddings
.:,2Adam/m/embedding_81/embeddings
.:,2Adam/v/embedding_81/embeddings
.:,2Adam/m/embedding_82/embeddings
.:,2Adam/v/embedding_82/embeddings
.:,2Adam/m/embedding_83/embeddings
.:,2Adam/v/embedding_83/embeddings
.:,2Adam/m/embedding_84/embeddings
.:,2Adam/v/embedding_84/embeddings
.:,2Adam/m/embedding_85/embeddings
.:,2Adam/v/embedding_85/embeddings
.:,2Adam/m/embedding_86/embeddings
.:,2Adam/v/embedding_86/embeddings
.:,2Adam/m/embedding_89/embeddings
.:,2Adam/v/embedding_89/embeddings
.:,W	2Adam/m/embedding_90/embeddings
.:,W	2Adam/v/embedding_90/embeddings
.:,2Adam/m/embedding_87/embeddings
.:,2Adam/v/embedding_87/embeddings
.:,2Adam/m/embedding_88/embeddings
.:,2Adam/v/embedding_88/embeddings
-:+	ђ2Adam/m/user_embedding/kernel
-:+	ђ2Adam/v/user_embedding/kernel
':%ђ2Adam/m/user_embedding/bias
':%ђ2Adam/v/user_embedding/bias
.:,
£ђ2Adam/m/food_embedding/kernel
.:,
£ђ2Adam/v/food_embedding/kernel
':%ђ2Adam/m/food_embedding/bias
':%ђ2Adam/v/food_embedding/bias
0:.	Ш2Adam/m/context_embedding/kernel
0:.	Ш2Adam/v/context_embedding/kernel
):'2Adam/m/context_embedding/bias
):'2Adam/v/context_embedding/bias
0:.й2#Adam/m/batch_normalization_12/gamma
0:.й2#Adam/v/batch_normalization_12/gamma
/:-й2"Adam/m/batch_normalization_12/beta
/:-й2"Adam/v/batch_normalization_12/beta
*:(
йА2Adam/m/fc_layer_0/kernel
*:(
йА2Adam/v/fc_layer_0/kernel
#:!А2Adam/m/fc_layer_0/bias
#:!А2Adam/v/fc_layer_0/bias
):'	А@2Adam/m/fc_layer_1/kernel
):'	А@2Adam/v/fc_layer_1/kernel
": @2Adam/m/fc_layer_1/bias
": @2Adam/v/fc_layer_1/bias
(:&@ 2Adam/m/fc_layer_2/kernel
(:&@ 2Adam/v/fc_layer_2/kernel
":  2Adam/m/fc_layer_2/bias
":  2Adam/v/fc_layer_2/bias
&:$ 2Adam/m/output_0/kernel
&:$ 2Adam/v/output_0/kernel
 :2Adam/m/output_0/bias
 :2Adam/v/output_0/bias
ЗBД
$__inference__update_step_xla_7096955gradient
gradient_1
gradient_2variable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЗBД
$__inference__update_step_xla_7096962gradient
gradient_1
gradient_2variable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЗBД
$__inference__update_step_xla_7096969gradient
gradient_1
gradient_2variable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЗBД
$__inference__update_step_xla_7096976gradient
gradient_1
gradient_2variable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЗBД
$__inference__update_step_xla_7096983gradient
gradient_1
gradient_2variable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЗBД
$__inference__update_step_xla_7096990gradient
gradient_1
gradient_2variable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЗBД
$__inference__update_step_xla_7096997gradient
gradient_1
gradient_2variable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЗBД
$__inference__update_step_xla_7097004gradient
gradient_1
gradient_2variable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЗBД
$__inference__update_step_xla_7097011gradient
gradient_1
gradient_2variable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЗBД
$__inference__update_step_xla_7097018gradient
gradient_1
gradient_2variable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЗBД
$__inference__update_step_xla_7097025gradient
gradient_1
gradient_2variable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЗBД
$__inference__update_step_xla_7097032gradient
gradient_1
gradient_2variable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЗBД
$__inference__update_step_xla_7097039gradient
gradient_1
gradient_2variable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЗBД
$__inference__update_step_xla_7097046gradient
gradient_1
gradient_2variable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЗBД
$__inference__update_step_xla_7097053gradient
gradient_1
gradient_2variable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
пBм
$__inference__update_step_xla_7097058gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
пBм
$__inference__update_step_xla_7097063gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
пBм
$__inference__update_step_xla_7097068gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
пBм
$__inference__update_step_xla_7097073gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
пBм
$__inference__update_step_xla_7097078gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
пBм
$__inference__update_step_xla_7097083gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
пBм
$__inference__update_step_xla_7097088gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
пBм
$__inference__update_step_xla_7097093gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
пBм
$__inference__update_step_xla_7097098gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
пBм
$__inference__update_step_xla_7097103gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
пBм
$__inference__update_step_xla_7097108gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
пBм
$__inference__update_step_xla_7097113gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
пBм
$__inference__update_step_xla_7097118gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
пBм
$__inference__update_step_xla_7097123gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
пBм
$__inference__update_step_xla_7097128gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
пBм
$__inference__update_step_xla_7097133gradientvariable"≠
¶≤Ґ
FullArgSpec*
args"Ъ

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
≥B∞
__inference__creator_7097964"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
ч
Е	capture_1
Ж	capture_2Bі
 __inference__initializer_7097971"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zЕ	capture_1zЖ	capture_2
µB≤
__inference__destroyer_7097975"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
≥B∞
__inference__creator_7097979"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
ч
З	capture_1
И	capture_2Bі
 __inference__initializer_7097986"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zЗ	capture_1zИ	capture_2
µB≤
__inference__destroyer_7097990"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
≥B∞
__inference__creator_7097994"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
ч
Й	capture_1
К	capture_2Bі
 __inference__initializer_7098001"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zЙ	capture_1zК	capture_2
µB≤
__inference__destroyer_7098005"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
≥B∞
__inference__creator_7098009"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
ч
Л	capture_1
М	capture_2Bі
 __inference__initializer_7098016"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zЛ	capture_1zМ	capture_2
µB≤
__inference__destroyer_7098020"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
≥B∞
__inference__creator_7098024"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
ч
Н	capture_1
О	capture_2Bі
 __inference__initializer_7098031"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zН	capture_1zО	capture_2
µB≤
__inference__destroyer_7098035"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
≥B∞
__inference__creator_7098039"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
ч
П	capture_1
Р	capture_2Bі
 __inference__initializer_7098046"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zП	capture_1zР	capture_2
µB≤
__inference__destroyer_7098050"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
≥B∞
__inference__creator_7098054"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
ч
С	capture_1
Т	capture_2Bі
 __inference__initializer_7098061"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zС	capture_1zТ	capture_2
µB≤
__inference__destroyer_7098065"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
≥B∞
__inference__creator_7098069"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
ч
У	capture_1
Ф	capture_2Bі
 __inference__initializer_7098076"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zУ	capture_1zФ	capture_2
µB≤
__inference__destroyer_7098080"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
≥B∞
__inference__creator_7098084"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
ч
Х	capture_1
Ц	capture_2Bі
 __inference__initializer_7098091"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zХ	capture_1zЦ	capture_2
µB≤
__inference__destroyer_7098095"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
≥B∞
__inference__creator_7098099"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
ч
Ч	capture_1
Ш	capture_2Bі
 __inference__initializer_7098106"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zЧ	capture_1zШ	capture_2
µB≤
__inference__destroyer_7098110"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
≥B∞
__inference__creator_7098114"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
ч
Щ	capture_1
Ъ	capture_2Bі
 __inference__initializer_7098121"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zЩ	capture_1zЪ	capture_2
µB≤
__inference__destroyer_7098125"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
≥B∞
__inference__creator_7098129"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
ч
Ы	capture_1
Ь	capture_2Bі
 __inference__initializer_7098136"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zЫ	capture_1zЬ	capture_2
µB≤
__inference__destroyer_7098140"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
≥B∞
__inference__creator_7098144"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
ч
Э	capture_1
Ю	capture_2Bі
 __inference__initializer_7098151"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zЭ	capture_1zЮ	capture_2
µB≤
__inference__destroyer_7098155"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
≥B∞
__inference__creator_7098159"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
ч
Я	capture_1
†	capture_2Bі
 __inference__initializer_7098166"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ zЯ	capture_1z†	capture_2
µB≤
__inference__destroyer_7098170"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
≥B∞
__inference__creator_7098174"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
ч
°	capture_1
Ґ	capture_2Bі
 __inference__initializer_7098181"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ z°	capture_1zҐ	capture_2
µB≤
__inference__destroyer_7098185"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
≥B∞
__inference__creator_7098189"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
ч
£	capture_1
§	capture_2Bі
 __inference__initializer_7098196"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ z£	capture_1z§	capture_2
µB≤
__inference__destroyer_7098200"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
≥B∞
__inference__creator_7098204"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
ч
•	capture_1
¶	capture_2Bі
 __inference__initializer_7098211"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ z•	capture_1z¶	capture_2
µB≤
__inference__destroyer_7098215"П
З≤Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *Ґ 
0
ф0
х1"
trackable_list_wrapper
.
т	variables"
_generic_user_object
:  (2total
:  (2count
0
ш0
щ1"
trackable_list_wrapper
.
ц	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
ю0
€1"
trackable_list_wrapper
.
ы	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_positives
0
Г0
Д1"
trackable_list_wrapper
.
А	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_negatives
"J

Const_33jtf.TrackableConstant
"J

Const_32jtf.TrackableConstant
"J

Const_30jtf.TrackableConstant
"J

Const_31jtf.TrackableConstant
"J

Const_29jtf.TrackableConstant
"J

Const_28jtf.TrackableConstant
"J

Const_27jtf.TrackableConstant
"J

Const_26jtf.TrackableConstant
"J

Const_25jtf.TrackableConstant
"J

Const_24jtf.TrackableConstant
"J

Const_22jtf.TrackableConstant
"J

Const_23jtf.TrackableConstant
"J

Const_21jtf.TrackableConstant
"J

Const_20jtf.TrackableConstant
"J

Const_19jtf.TrackableConstant
"J

Const_18jtf.TrackableConstant
"J

Const_17jtf.TrackableConstant
"J

Const_16jtf.TrackableConstant
"J

Const_14jtf.TrackableConstant
"J

Const_15jtf.TrackableConstant
"J

Const_13jtf.TrackableConstant
"J

Const_12jtf.TrackableConstant
"J

Const_11jtf.TrackableConstant
"J

Const_10jtf.TrackableConstant
!J	
Const_9jtf.TrackableConstant
!J	
Const_8jtf.TrackableConstant
!J	
Const_6jtf.TrackableConstant
!J	
Const_7jtf.TrackableConstant
!J	
Const_5jtf.TrackableConstant
!J	
Const_4jtf.TrackableConstant
!J	
Const_3jtf.TrackableConstant
!J	
Const_2jtf.TrackableConstant
!J	
Const_1jtf.TrackableConstant
J
Constjtf.TrackableConstantA
__inference__creator_7097964!Ґ

Ґ 
™ "К
unknown A
__inference__creator_7097979!Ґ

Ґ 
™ "К
unknown A
__inference__creator_7097994!Ґ

Ґ 
™ "К
unknown A
__inference__creator_7098009!Ґ

Ґ 
™ "К
unknown A
__inference__creator_7098024!Ґ

Ґ 
™ "К
unknown A
__inference__creator_7098039!Ґ

Ґ 
™ "К
unknown A
__inference__creator_7098054!Ґ

Ґ 
™ "К
unknown A
__inference__creator_7098069!Ґ

Ґ 
™ "К
unknown A
__inference__creator_7098084!Ґ

Ґ 
™ "К
unknown A
__inference__creator_7098099!Ґ

Ґ 
™ "К
unknown A
__inference__creator_7098114!Ґ

Ґ 
™ "К
unknown A
__inference__creator_7098129!Ґ

Ґ 
™ "К
unknown A
__inference__creator_7098144!Ґ

Ґ 
™ "К
unknown A
__inference__creator_7098159!Ґ

Ґ 
™ "К
unknown A
__inference__creator_7098174!Ґ

Ґ 
™ "К
unknown A
__inference__creator_7098189!Ґ

Ґ 
™ "К
unknown A
__inference__creator_7098204!Ґ

Ґ 
™ "К
unknown C
__inference__destroyer_7097975!Ґ

Ґ 
™ "К
unknown C
__inference__destroyer_7097990!Ґ

Ґ 
™ "К
unknown C
__inference__destroyer_7098005!Ґ

Ґ 
™ "К
unknown C
__inference__destroyer_7098020!Ґ

Ґ 
™ "К
unknown C
__inference__destroyer_7098035!Ґ

Ґ 
™ "К
unknown C
__inference__destroyer_7098050!Ґ

Ґ 
™ "К
unknown C
__inference__destroyer_7098065!Ґ

Ґ 
™ "К
unknown C
__inference__destroyer_7098080!Ґ

Ґ 
™ "К
unknown C
__inference__destroyer_7098095!Ґ

Ґ 
™ "К
unknown C
__inference__destroyer_7098110!Ґ

Ґ 
™ "К
unknown C
__inference__destroyer_7098125!Ґ

Ґ 
™ "К
unknown C
__inference__destroyer_7098140!Ґ

Ґ 
™ "К
unknown C
__inference__destroyer_7098155!Ґ

Ґ 
™ "К
unknown C
__inference__destroyer_7098170!Ґ

Ґ 
™ "К
unknown C
__inference__destroyer_7098185!Ґ

Ґ 
™ "К
unknown C
__inference__destroyer_7098200!Ґ

Ґ 
™ "К
unknown C
__inference__destroyer_7098215!Ґ

Ґ 
™ "К
unknown L
 __inference__initializer_7097971(fЕЖҐ

Ґ 
™ "К
unknown L
 __inference__initializer_7097986(iЗИҐ

Ґ 
™ "К
unknown L
 __inference__initializer_7098001(lЙКҐ

Ґ 
™ "К
unknown L
 __inference__initializer_7098016(oЛМҐ

Ґ 
™ "К
unknown L
 __inference__initializer_7098031(rНОҐ

Ґ 
™ "К
unknown L
 __inference__initializer_7098046(uПРҐ

Ґ 
™ "К
unknown L
 __inference__initializer_7098061(xСТҐ

Ґ 
™ "К
unknown L
 __inference__initializer_7098076({УФҐ

Ґ 
™ "К
unknown L
 __inference__initializer_7098091(~ХЦҐ

Ґ 
™ "К
unknown M
 __inference__initializer_7098106)БЧШҐ

Ґ 
™ "К
unknown M
 __inference__initializer_7098121)ДЩЪҐ

Ґ 
™ "К
unknown M
 __inference__initializer_7098136)ЗЫЬҐ

Ґ 
™ "К
unknown M
 __inference__initializer_7098151)КЭЮҐ

Ґ 
™ "К
unknown M
 __inference__initializer_7098166)иЯ†Ґ

Ґ 
™ "К
unknown M
 __inference__initializer_7098181)л°ҐҐ

Ґ 
™ "К
unknown M
 __inference__initializer_7098196)Љ£§Ґ

Ґ 
™ "К
unknown M
 __inference__initializer_7098211)ў•¶Ґ

Ґ 
™ "К
unknown Ў
$__inference__update_step_xla_7096955ѓ®Ґ§
ЬҐШ
VТS:Ґ7
ъ€€€€€€€€€
А
А	
А
ъ€€€€€€€€€IndexedSlicesSpec 
4Т1	Ґ
ъ
А
p
` VariableSpec 
`аЏЇЬАо=
™ "
 Ў
$__inference__update_step_xla_7096962ѓ®Ґ§
ЬҐШ
VТS:Ґ7
ъ€€€€€€€€€
А
А	
А
ъ€€€€€€€€€IndexedSlicesSpec 
4Т1	Ґ
ъ
А
p
` VariableSpec 
`АчЛХАо=
™ "
 Ў
$__inference__update_step_xla_7096969ѓ®Ґ§
ЬҐШ
VТS:Ґ7
ъ€€€€€€€€€
А
А	
А
ъ€€€€€€€€€IndexedSlicesSpec 
4Т1	Ґ
ъ
А
p
` VariableSpec 
`јаХДПо=
™ "
 Ў
$__inference__update_step_xla_7096976ѓ®Ґ§
ЬҐШ
VТS:Ґ7
ъ€€€€€€€€€
А
А	
А
ъ€€€€€€€€€IndexedSlicesSpec 
4Т1	Ґ
ъ
А
p
` VariableSpec 
`јчФДПо=
™ "
 Ў
$__inference__update_step_xla_7096983ѓ®Ґ§
ЬҐШ
VТS:Ґ7
ъ€€€€€€€€€
А
А	
А
ъ€€€€€€€€€IndexedSlicesSpec 
4Т1	Ґ
ъ
А
p
` VariableSpec 
`†уХДПо=
™ "
 Ў
$__inference__update_step_xla_7096990ѓ®Ґ§
ЬҐШ
VТS:Ґ7
ъ€€€€€€€€€
А
А	
А
ъ€€€€€€€€€IndexedSlicesSpec 
4Т1	Ґ
ъ
А
p
` VariableSpec 
`јЙФДПо=
™ "
 Ў
$__inference__update_step_xla_7096997ѓ®Ґ§
ЬҐШ
VТS:Ґ7
ъ€€€€€€€€€
А
А	
А
ъ€€€€€€€€€IndexedSlicesSpec 
4Т1	Ґ
ъ
А
p
` VariableSpec 
`АЫЎЗЖо=
™ "
 Ў
$__inference__update_step_xla_7097004ѓ®Ґ§
ЬҐШ
VТS:Ґ7
ъ€€€€€€€€€
А
А	
А
ъ€€€€€€€€€IndexedSlicesSpec 
4Т1	Ґ
ъ
А
p
` VariableSpec 
`аѓ•ЗПо=
™ "
 Ў
$__inference__update_step_xla_7097011ѓ®Ґ§
ЬҐШ
VТS:Ґ7
ъ€€€€€€€€€
А
А	
А
ъ€€€€€€€€€IndexedSlicesSpec 
4Т1	Ґ
ъ
А
p
` VariableSpec 
`а»µ«Но=
™ "
 Ў
$__inference__update_step_xla_7097018ѓ®Ґ§
ЬҐШ
VТS:Ґ7
ъ€€€€€€€€€
А
А	
А
ъ€€€€€€€€€IndexedSlicesSpec 
4Т1	Ґ
ъ
А
p
` VariableSpec 
`Аґµ«Но=
™ "
 Ў
$__inference__update_step_xla_7097025ѓ®Ґ§
ЬҐШ
VТS:Ґ7
ъ€€€€€€€€€
А
А	
А
ъ€€€€€€€€€IndexedSlicesSpec 
4Т1	Ґ
ъ
А
p
` VariableSpec 
`А€ЎЗЖо=
™ "
 Ў
$__inference__update_step_xla_7097032ѓ®Ґ§
ЬҐШ
VТS:Ґ7
ъ€€€€€€€€€
А
А	
А
ъ€€€€€€€€€IndexedSlicesSpec 
4Т1	Ґ
ъ
А
p
` VariableSpec 
`Аъ†ГЖо=
™ "
 Ў
$__inference__update_step_xla_7097039ѓ®Ґ§
ЬҐШ
VТS:Ґ7
ъ€€€€€€€€€	
А
А	
А
ъ€€€€€€€€€IndexedSlicesSpec 
4Т1	Ґ
ъW	
А
p
` VariableSpec 
`јљЧ™Ао=
™ "
 Ў
$__inference__update_step_xla_7097046ѓ®Ґ§
ЬҐШ
VТS:Ґ7
ъ€€€€€€€€€
А
А	
А
ъ€€€€€€€€€IndexedSlicesSpec 
4Т1	Ґ
ъ
А
p
` VariableSpec 
`аАЄВПо=
™ "
 Ў
$__inference__update_step_xla_7097053ѓ®Ґ§
ЬҐШ
VТS:Ґ7
ъ€€€€€€€€€
А
А	
А
ъ€€€€€€€€€IndexedSlicesSpec 
4Т1	Ґ
ъ
А
p
` VariableSpec 
`а∆ЄВПо=
™ "
 Ш
$__inference__update_step_xla_7097058pjҐg
`Ґ]
К
gradient	ђ
5Т2	Ґ
ъ	ђ
А
p
` VariableSpec 
`јЯљђАо=
™ "
 Р
$__inference__update_step_xla_7097063hbҐ_
XҐU
К
gradientђ
1Т.	Ґ
ъђ
А
p
` VariableSpec 
`ј÷љђАо=
™ "
 Ъ
$__inference__update_step_xla_7097068rlҐi
bҐ_
К
gradient
£ђ
6Т3	Ґ
ъ
£ђ
А
p
` VariableSpec 
`аЌ’ЧАо=
™ "
 Р
$__inference__update_step_xla_7097073hbҐ_
XҐU
К
gradientђ
1Т.	Ґ
ъђ
А
p
` VariableSpec 
`†∆√БЖо=
™ "
 Ш
$__inference__update_step_xla_7097078pjҐg
`Ґ]
К
gradient	Ш
5Т2	Ґ
ъ	Ш
А
p
` VariableSpec 
`ацГ≈Зо=
™ "
 О
$__inference__update_step_xla_7097083f`Ґ]
VҐS
К
gradient
0Т-	Ґ
ъ
А
p
` VariableSpec 
`аЮ§ЦАо=
™ "
 Р
$__inference__update_step_xla_7097088hbҐ_
XҐU
К
gradientй
1Т.	Ґ
ъй
А
p
` VariableSpec 
`а±©БЖо=
™ "
 Р
$__inference__update_step_xla_7097093hbҐ_
XҐU
К
gradientй
1Т.	Ґ
ъй
А
p
` VariableSpec 
`†ф®БЖо=
™ "
 Ъ
$__inference__update_step_xla_7097098rlҐi
bҐ_
К
gradient
йА
6Т3	Ґ
ъ
йА
А
p
` VariableSpec 
`АґЙ√Зо=
™ "
 Р
$__inference__update_step_xla_7097103hbҐ_
XҐU
К
gradientА
1Т.	Ґ
ъА
А
p
` VariableSpec 
`†оЙ√Зо=
™ "
 Ш
$__inference__update_step_xla_7097108pjҐg
`Ґ]
К
gradient	А@
5Т2	Ґ
ъ	А@
А
p
` VariableSpec 
`а„Й√Зо=
™ "
 О
$__inference__update_step_xla_7097113f`Ґ]
VҐS
К
gradient@
0Т-	Ґ
ъ@
А
p
` VariableSpec 
`†ЂИ√Зо=
™ "
 Ц
$__inference__update_step_xla_7097118nhҐe
^Ґ[
К
gradient@ 
4Т1	Ґ
ъ@ 
А
p
` VariableSpec 
`А»И√Зо=
™ "
 О
$__inference__update_step_xla_7097123f`Ґ]
VҐS
К
gradient 
0Т-	Ґ
ъ 
А
p
` VariableSpec 
`јґИ√Зо=
™ "
 Ц
$__inference__update_step_xla_7097128nhҐe
^Ґ[
К
gradient 
4Т1	Ґ
ъ 
А
p
` VariableSpec 
`†ЁИ√Зо=
™ "
 О
$__inference__update_step_xla_7097133f`Ґ]
VҐS
К
gradient
0Т-	Ґ
ъ
А
p
` VariableSpec 
`аѓЙ√Зо=
™ "
 §
"__inference__wrapped_model_7094806э}КЋЗћДЌБќ~ѕ{–x—u“r”o‘l’i÷f„лЎиўеё„–…¬їі≠¶ЯШС √ЉЏўџмнфхВГЪЧЩШ°Ґ©™±≤єЇ∆Ґ¬
ЇҐґ
≥™ѓ
$
BMIК
BMI€€€€€€€€€
0
	age_range#К 
	age_range€€€€€€€€€
0
	allergens#К 
	allergens€€€€€€€€€
,
allergy!К
allergy€€€€€€€€€
.
calories"К
calories€€€€€€€€€
8
carbohydrates'К$
carbohydrates€€€€€€€€€
<
clinical_gender)К&
clinical_gender€€€€€€€€€
<
cultural_factor)К&
cultural_factor€€€€€€€€€
F
cultural_restriction.К+
cultural_restriction€€€€€€€€€
J
current_daily_calories0К-
current_daily_calories€€€€€€€€€
J
current_working_status0К-
current_working_status€€€€€€€€€
2

day_number$К!

day_number€€€€€€€€€
3

embeddings%К"

embeddings€€€€€€€€€А
0
	ethnicity#К 
	ethnicity€€€€€€€€€
$
fatК
fat€€€€€€€€€
(
fiberК
fiber€€€€€€€€€
*
height К
height€€€€€€€€€
2

life_style$К!

life_style€€€€€€€€€
:
marital_status(К%
marital_status€€€€€€€€€
4
meal_type_y%К"
meal_type_y€€€€€€€€€
.
next_BMI"К
next_BMI€€€€€€€€€
:
nutrition_goal(К%
nutrition_goal€€€€€€€€€
P
place_of_meal_consumption3К0
place_of_meal_consumption€€€€€€€€€
(
priceК
price€€€€€€€€€
N
projected_daily_calories2К/
projected_daily_calories€€€€€€€€€
,
protein!К
protein€€€€€€€€€
f
$social_situation_of_meal_consumption>К;
$social_situation_of_meal_consumption€€€€€€€€€
(
tasteК
taste€€€€€€€€€
N
time_of_meal_consumption2К/
time_of_meal_consumption€€€€€€€€€
*
weight К
weight€€€€€€€€€
™ "3™0
.
output_0"К
output_0€€€€€€€€€ 
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_7097788sЩЪЧШ8Ґ5
.Ґ+
!К
inputs€€€€€€€€€й
p

 
™ "-Ґ*
#К 
tensor_0€€€€€€€€€й
Ъ  
S__inference_batch_normalization_12_layer_call_and_return_conditional_losses_7097808sЪЧЩШ8Ґ5
.Ґ+
!К
inputs€€€€€€€€€й
p 

 
™ "-Ґ*
#К 
tensor_0€€€€€€€€€й
Ъ §
8__inference_batch_normalization_12_layer_call_fn_7097741hЩЪЧШ8Ґ5
.Ґ+
!К
inputs€€€€€€€€€й
p

 
™ ""К
unknown€€€€€€€€€й§
8__inference_batch_normalization_12_layer_call_fn_7097754hЪЧЩШ8Ґ5
.Ґ+
!К
inputs€€€€€€€€€й
p 

 
™ ""К
unknown€€€€€€€€€йє
K__inference_concatenate_31_layer_call_and_return_conditional_losses_7097540йЄҐі
ђҐ®
•Ъ°
"К
inputs_0€€€€€€€€€
"К
inputs_1€€€€€€€€€
"К
inputs_2€€€€€€€€€
"К
inputs_3€€€€€€€€€
"К
inputs_4€€€€€€€€€
"К
inputs_5€€€€€€€€€
"К
inputs_6€€€€€€€€€
"К
inputs_7€€€€€€€€€
"К
inputs_8€€€€€€€€€
"К
inputs_9€€€€€€€€€
#К 
	inputs_10€€€€€€€€€
#К 
	inputs_11€€€€€€€€€
#К 
	inputs_12€€€€€€€€€
#К 
	inputs_13€€€€€€€€€
#К 
	inputs_14€€€€€€€€€
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ У
0__inference_concatenate_31_layer_call_fn_7097520ёЄҐі
ђҐ®
•Ъ°
"К
inputs_0€€€€€€€€€
"К
inputs_1€€€€€€€€€
"К
inputs_2€€€€€€€€€
"К
inputs_3€€€€€€€€€
"К
inputs_4€€€€€€€€€
"К
inputs_5€€€€€€€€€
"К
inputs_6€€€€€€€€€
"К
inputs_7€€€€€€€€€
"К
inputs_8€€€€€€€€€
"К
inputs_9€€€€€€€€€
#К 
	inputs_10€€€€€€€€€
#К 
	inputs_11€€€€€€€€€
#К 
	inputs_12€€€€€€€€€
#К 
	inputs_13€€€€€€€€€
#К 
	inputs_14€€€€€€€€€
™ "!К
unknown€€€€€€€€€ќ
K__inference_concatenate_32_layer_call_and_return_conditional_losses_7097656юћҐ»
јҐЉ
єЪµ
"К
inputs_0€€€€€€€€€
#К 
inputs_1€€€€€€€€€У
"К
inputs_2€€€€€€€€€
"К
inputs_3€€€€€€€€€
"К
inputs_4€€€€€€€€€
™ "-Ґ*
#К 
tensor_0€€€€€€€€€Ш
Ъ ®
0__inference_concatenate_32_layer_call_fn_7097646ућҐ»
јҐЉ
єЪµ
"К
inputs_0€€€€€€€€€
#К 
inputs_1€€€€€€€€€У
"К
inputs_2€€€€€€€€€
"К
inputs_3€€€€€€€€€
"К
inputs_4€€€€€€€€€
™ ""К
unknown€€€€€€€€€ШВ
K__inference_concatenate_33_layer_call_and_return_conditional_losses_7097569≤АҐь
фҐр
нЪй
"К
inputs_0€€€€€€€€€
"К
inputs_1€€€€€€€€€
"К
inputs_2€€€€€€€€€	
"К
inputs_3€€€€€€€€€
"К
inputs_4€€€€€€€€€
"К
inputs_5€€€€€€€€€
"К
inputs_6€€€€€€€€€
"К
inputs_7€€€€€€€€€
"К
inputs_8€€€€€€€€€
#К 
inputs_9€€€€€€€€€А
™ "-Ґ*
#К 
tensor_0€€€€€€€€€£
Ъ №
0__inference_concatenate_33_layer_call_fn_7097554ІАҐь
фҐр
нЪй
"К
inputs_0€€€€€€€€€
"К
inputs_1€€€€€€€€€
"К
inputs_2€€€€€€€€€	
"К
inputs_3€€€€€€€€€
"К
inputs_4€€€€€€€€€
"К
inputs_5€€€€€€€€€
"К
inputs_6€€€€€€€€€
"К
inputs_7€€€€€€€€€
"К
inputs_8€€€€€€€€€
#К 
inputs_9€€€€€€€€€А
™ ""К
unknown€€€€€€€€€£Ђ
K__inference_concatenate_34_layer_call_and_return_conditional_losses_7097728џ©Ґ•
ЭҐЩ
ЦЪТ
#К 
inputs_0€€€€€€€€€ђ
#К 
inputs_1€€€€€€€€€ђ
"К
inputs_2€€€€€€€€€
"К
inputs_3€€€€€€€€€
™ "-Ґ*
#К 
tensor_0€€€€€€€€€й
Ъ Е
0__inference_concatenate_34_layer_call_fn_7097719–©Ґ•
ЭҐЩ
ЦЪТ
#К 
inputs_0€€€€€€€€€ђ
#К 
inputs_1€€€€€€€€€ђ
"К
inputs_2€€€€€€€€€
"К
inputs_3€€€€€€€€€
™ ""К
unknown€€€€€€€€€йЄ
N__inference_context_embedding_layer_call_and_return_conditional_losses_7097679fВГ0Ґ-
&Ґ#
!К
inputs€€€€€€€€€Ш
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ Т
3__inference_context_embedding_layer_call_fn_7097665[ВГ0Ґ-
&Ґ#
!К
inputs€€€€€€€€€Ш
™ "!К
unknown€€€€€€€€€‘
C__inference_dot_12_layer_call_and_return_conditional_losses_7097711М\ҐY
RҐO
MЪJ
#К 
inputs_0€€€€€€€€€ђ
#К 
inputs_1€€€€€€€€€ђ
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ Ѓ
(__inference_dot_12_layer_call_fn_7097685Б\ҐY
RҐO
MЪJ
#К 
inputs_0€€€€€€€€€ђ
#К 
inputs_1€€€€€€€€€ђ
™ "!К
unknown€€€€€€€€€і
I__inference_embedding_76_layer_call_and_return_conditional_losses_7097148gС/Ґ,
%Ґ"
 К
inputs€€€€€€€€€	
™ "0Ґ-
&К#
tensor_0€€€€€€€€€
Ъ О
.__inference_embedding_76_layer_call_fn_7097140\С/Ґ,
%Ґ"
 К
inputs€€€€€€€€€	
™ "%К"
unknown€€€€€€€€€і
I__inference_embedding_77_layer_call_and_return_conditional_losses_7097163gШ/Ґ,
%Ґ"
 К
inputs€€€€€€€€€	
™ "0Ґ-
&К#
tensor_0€€€€€€€€€
Ъ О
.__inference_embedding_77_layer_call_fn_7097155\Ш/Ґ,
%Ґ"
 К
inputs€€€€€€€€€	
™ "%К"
unknown€€€€€€€€€і
I__inference_embedding_78_layer_call_and_return_conditional_losses_7097178gЯ/Ґ,
%Ґ"
 К
inputs€€€€€€€€€	
™ "0Ґ-
&К#
tensor_0€€€€€€€€€
Ъ О
.__inference_embedding_78_layer_call_fn_7097170\Я/Ґ,
%Ґ"
 К
inputs€€€€€€€€€	
™ "%К"
unknown€€€€€€€€€і
I__inference_embedding_79_layer_call_and_return_conditional_losses_7097193g¶/Ґ,
%Ґ"
 К
inputs€€€€€€€€€	
™ "0Ґ-
&К#
tensor_0€€€€€€€€€
Ъ О
.__inference_embedding_79_layer_call_fn_7097185\¶/Ґ,
%Ґ"
 К
inputs€€€€€€€€€	
™ "%К"
unknown€€€€€€€€€і
I__inference_embedding_80_layer_call_and_return_conditional_losses_7097208g≠/Ґ,
%Ґ"
 К
inputs€€€€€€€€€	
™ "0Ґ-
&К#
tensor_0€€€€€€€€€
Ъ О
.__inference_embedding_80_layer_call_fn_7097200\≠/Ґ,
%Ґ"
 К
inputs€€€€€€€€€	
™ "%К"
unknown€€€€€€€€€і
I__inference_embedding_81_layer_call_and_return_conditional_losses_7097223gі/Ґ,
%Ґ"
 К
inputs€€€€€€€€€	
™ "0Ґ-
&К#
tensor_0€€€€€€€€€
Ъ О
.__inference_embedding_81_layer_call_fn_7097215\і/Ґ,
%Ґ"
 К
inputs€€€€€€€€€	
™ "%К"
unknown€€€€€€€€€і
I__inference_embedding_82_layer_call_and_return_conditional_losses_7097238gї/Ґ,
%Ґ"
 К
inputs€€€€€€€€€	
™ "0Ґ-
&К#
tensor_0€€€€€€€€€
Ъ О
.__inference_embedding_82_layer_call_fn_7097230\ї/Ґ,
%Ґ"
 К
inputs€€€€€€€€€	
™ "%К"
unknown€€€€€€€€€і
I__inference_embedding_83_layer_call_and_return_conditional_losses_7097253g¬/Ґ,
%Ґ"
 К
inputs€€€€€€€€€	
™ "0Ґ-
&К#
tensor_0€€€€€€€€€
Ъ О
.__inference_embedding_83_layer_call_fn_7097245\¬/Ґ,
%Ґ"
 К
inputs€€€€€€€€€	
™ "%К"
unknown€€€€€€€€€і
I__inference_embedding_84_layer_call_and_return_conditional_losses_7097268g…/Ґ,
%Ґ"
 К
inputs€€€€€€€€€	
™ "0Ґ-
&К#
tensor_0€€€€€€€€€
Ъ О
.__inference_embedding_84_layer_call_fn_7097260\…/Ґ,
%Ґ"
 К
inputs€€€€€€€€€	
™ "%К"
unknown€€€€€€€€€і
I__inference_embedding_85_layer_call_and_return_conditional_losses_7097283g–/Ґ,
%Ґ"
 К
inputs€€€€€€€€€	
™ "0Ґ-
&К#
tensor_0€€€€€€€€€
Ъ О
.__inference_embedding_85_layer_call_fn_7097275\–/Ґ,
%Ґ"
 К
inputs€€€€€€€€€	
™ "%К"
unknown€€€€€€€€€і
I__inference_embedding_86_layer_call_and_return_conditional_losses_7097298g„/Ґ,
%Ґ"
 К
inputs€€€€€€€€€	
™ "0Ґ-
&К#
tensor_0€€€€€€€€€
Ъ О
.__inference_embedding_86_layer_call_fn_7097290\„/Ґ,
%Ґ"
 К
inputs€€€€€€€€€	
™ "%К"
unknown€€€€€€€€€і
I__inference_embedding_87_layer_call_and_return_conditional_losses_7097486g√/Ґ,
%Ґ"
 К
inputs€€€€€€€€€	
™ "0Ґ-
&К#
tensor_0€€€€€€€€€
Ъ О
.__inference_embedding_87_layer_call_fn_7097478\√/Ґ,
%Ґ"
 К
inputs€€€€€€€€€	
™ "%К"
unknown€€€€€€€€€і
I__inference_embedding_88_layer_call_and_return_conditional_losses_7097501g /Ґ,
%Ґ"
 К
inputs€€€€€€€€€	
™ "0Ґ-
&К#
tensor_0€€€€€€€€€
Ъ О
.__inference_embedding_88_layer_call_fn_7097493\ /Ґ,
%Ґ"
 К
inputs€€€€€€€€€	
™ "%К"
unknown€€€€€€€€€і
I__inference_embedding_89_layer_call_and_return_conditional_losses_7097313gё/Ґ,
%Ґ"
 К
inputs€€€€€€€€€	
™ "0Ґ-
&К#
tensor_0€€€€€€€€€
Ъ О
.__inference_embedding_89_layer_call_fn_7097305\ё/Ґ,
%Ґ"
 К
inputs€€€€€€€€€	
™ "%К"
unknown€€€€€€€€€і
I__inference_embedding_90_layer_call_and_return_conditional_losses_7097328gе/Ґ,
%Ґ"
 К
inputs€€€€€€€€€	
™ "0Ґ-
&К#
tensor_0€€€€€€€€€	
Ъ О
.__inference_embedding_90_layer_call_fn_7097320\е/Ґ,
%Ґ"
 К
inputs€€€€€€€€€	
™ "%К"
unknown€€€€€€€€€	≤
G__inference_fc_layer_0_layer_call_and_return_conditional_losses_7097832g°Ґ0Ґ-
&Ґ#
!К
inputs€€€€€€€€€й
™ "-Ґ*
#К 
tensor_0€€€€€€€€€А
Ъ М
,__inference_fc_layer_0_layer_call_fn_7097817\°Ґ0Ґ-
&Ґ#
!К
inputs€€€€€€€€€й
™ ""К
unknown€€€€€€€€€А±
G__inference_fc_layer_1_layer_call_and_return_conditional_losses_7097856f©™0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ ",Ґ)
"К
tensor_0€€€€€€€€€@
Ъ Л
,__inference_fc_layer_1_layer_call_fn_7097841[©™0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "!К
unknown€€€€€€€€€@∞
G__inference_fc_layer_2_layer_call_and_return_conditional_losses_7097880e±≤/Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ ",Ґ)
"К
tensor_0€€€€€€€€€ 
Ъ К
,__inference_fc_layer_2_layer_call_fn_7097865Z±≤/Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "!К
unknown€€€€€€€€€ Ѓ
G__inference_flatten_76_layer_call_and_return_conditional_losses_7097339c3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ И
,__inference_flatten_76_layer_call_fn_7097333X3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€
™ "!К
unknown€€€€€€€€€Ѓ
G__inference_flatten_77_layer_call_and_return_conditional_losses_7097350c3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ И
,__inference_flatten_77_layer_call_fn_7097344X3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€
™ "!К
unknown€€€€€€€€€Ѓ
G__inference_flatten_78_layer_call_and_return_conditional_losses_7097361c3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ И
,__inference_flatten_78_layer_call_fn_7097355X3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€
™ "!К
unknown€€€€€€€€€Ѓ
G__inference_flatten_79_layer_call_and_return_conditional_losses_7097372c3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ И
,__inference_flatten_79_layer_call_fn_7097366X3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€
™ "!К
unknown€€€€€€€€€Ѓ
G__inference_flatten_80_layer_call_and_return_conditional_losses_7097383c3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ И
,__inference_flatten_80_layer_call_fn_7097377X3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€
™ "!К
unknown€€€€€€€€€Ѓ
G__inference_flatten_81_layer_call_and_return_conditional_losses_7097394c3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ И
,__inference_flatten_81_layer_call_fn_7097388X3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€
™ "!К
unknown€€€€€€€€€Ѓ
G__inference_flatten_82_layer_call_and_return_conditional_losses_7097405c3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ И
,__inference_flatten_82_layer_call_fn_7097399X3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€
™ "!К
unknown€€€€€€€€€Ѓ
G__inference_flatten_83_layer_call_and_return_conditional_losses_7097416c3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ И
,__inference_flatten_83_layer_call_fn_7097410X3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€
™ "!К
unknown€€€€€€€€€Ѓ
G__inference_flatten_84_layer_call_and_return_conditional_losses_7097427c3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ И
,__inference_flatten_84_layer_call_fn_7097421X3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€
™ "!К
unknown€€€€€€€€€Ѓ
G__inference_flatten_85_layer_call_and_return_conditional_losses_7097438c3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ И
,__inference_flatten_85_layer_call_fn_7097432X3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€
™ "!К
unknown€€€€€€€€€Ѓ
G__inference_flatten_86_layer_call_and_return_conditional_losses_7097449c3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ И
,__inference_flatten_86_layer_call_fn_7097443X3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€
™ "!К
unknown€€€€€€€€€Ѓ
G__inference_flatten_87_layer_call_and_return_conditional_losses_7097580c3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ И
,__inference_flatten_87_layer_call_fn_7097574X3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€
™ "!К
unknown€€€€€€€€€Ѓ
G__inference_flatten_88_layer_call_and_return_conditional_losses_7097591c3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ И
,__inference_flatten_88_layer_call_fn_7097585X3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€
™ "!К
unknown€€€€€€€€€Ѓ
G__inference_flatten_89_layer_call_and_return_conditional_losses_7097460c3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ И
,__inference_flatten_89_layer_call_fn_7097454X3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€
™ "!К
unknown€€€€€€€€€Ѓ
G__inference_flatten_90_layer_call_and_return_conditional_losses_7097471c3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€	
™ ",Ґ)
"К
tensor_0€€€€€€€€€	
Ъ И
,__inference_flatten_90_layer_call_fn_7097465X3Ґ0
)Ґ&
$К!
inputs€€€€€€€€€	
™ "!К
unknown€€€€€€€€€	ґ
K__inference_food_embedding_layer_call_and_return_conditional_losses_7097637gфх0Ґ-
&Ґ#
!К
inputs€€€€€€€€€£
™ "-Ґ*
#К 
tensor_0€€€€€€€€€ђ
Ъ Р
0__inference_food_embedding_layer_call_fn_7097623\фх0Ґ-
&Ґ#
!К
inputs€€€€€€€€€£
™ ""К
unknown€€€€€€€€€ђF
__inference_loss_fn_0_7097912%мҐ

Ґ 
™ "К
unknown F
__inference_loss_fn_1_7097920%фҐ

Ґ 
™ "К
unknown F
__inference_loss_fn_2_7097928%ВҐ

Ґ 
™ "К
unknown F
__inference_loss_fn_3_7097936%°Ґ

Ґ 
™ "К
unknown F
__inference_loss_fn_4_7097944%©Ґ

Ґ 
™ "К
unknown F
__inference_loss_fn_5_7097952%±Ґ

Ґ 
™ "К
unknown F
__inference_loss_fn_6_7097960%єҐ

Ґ 
™ "К
unknown »
E__inference_model_12_layer_call_and_return_conditional_losses_7095682ю}КЋЗћДЌБќ~ѕ{–x—u“r”o‘l’i÷f„лЎиўеё„–…¬їі≠¶ЯШС √ЉЏўџмнфхВГЩЪЧШ°Ґ©™±≤єЇќҐ 
¬ҐЊ
≥™ѓ
$
BMIК
BMI€€€€€€€€€
0
	age_range#К 
	age_range€€€€€€€€€
0
	allergens#К 
	allergens€€€€€€€€€
,
allergy!К
allergy€€€€€€€€€
.
calories"К
calories€€€€€€€€€
8
carbohydrates'К$
carbohydrates€€€€€€€€€
<
clinical_gender)К&
clinical_gender€€€€€€€€€
<
cultural_factor)К&
cultural_factor€€€€€€€€€
F
cultural_restriction.К+
cultural_restriction€€€€€€€€€
J
current_daily_calories0К-
current_daily_calories€€€€€€€€€
J
current_working_status0К-
current_working_status€€€€€€€€€
2

day_number$К!

day_number€€€€€€€€€
3

embeddings%К"

embeddings€€€€€€€€€А
0
	ethnicity#К 
	ethnicity€€€€€€€€€
$
fatК
fat€€€€€€€€€
(
fiberК
fiber€€€€€€€€€
*
height К
height€€€€€€€€€
2

life_style$К!

life_style€€€€€€€€€
:
marital_status(К%
marital_status€€€€€€€€€
4
meal_type_y%К"
meal_type_y€€€€€€€€€
.
next_BMI"К
next_BMI€€€€€€€€€
:
nutrition_goal(К%
nutrition_goal€€€€€€€€€
P
place_of_meal_consumption3К0
place_of_meal_consumption€€€€€€€€€
(
priceК
price€€€€€€€€€
N
projected_daily_calories2К/
projected_daily_calories€€€€€€€€€
,
protein!К
protein€€€€€€€€€
f
$social_situation_of_meal_consumption>К;
$social_situation_of_meal_consumption€€€€€€€€€
(
tasteК
taste€€€€€€€€€
N
time_of_meal_consumption2К/
time_of_meal_consumption€€€€€€€€€
*
weight К
weight€€€€€€€€€
p

 
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ »
E__inference_model_12_layer_call_and_return_conditional_losses_7096086ю}КЋЗћДЌБќ~ѕ{–x—u“r”o‘l’i÷f„лЎиўеё„–…¬їі≠¶ЯШС √ЉЏўџмнфхВГЪЧЩШ°Ґ©™±≤єЇќҐ 
¬ҐЊ
≥™ѓ
$
BMIК
BMI€€€€€€€€€
0
	age_range#К 
	age_range€€€€€€€€€
0
	allergens#К 
	allergens€€€€€€€€€
,
allergy!К
allergy€€€€€€€€€
.
calories"К
calories€€€€€€€€€
8
carbohydrates'К$
carbohydrates€€€€€€€€€
<
clinical_gender)К&
clinical_gender€€€€€€€€€
<
cultural_factor)К&
cultural_factor€€€€€€€€€
F
cultural_restriction.К+
cultural_restriction€€€€€€€€€
J
current_daily_calories0К-
current_daily_calories€€€€€€€€€
J
current_working_status0К-
current_working_status€€€€€€€€€
2

day_number$К!

day_number€€€€€€€€€
3

embeddings%К"

embeddings€€€€€€€€€А
0
	ethnicity#К 
	ethnicity€€€€€€€€€
$
fatК
fat€€€€€€€€€
(
fiberК
fiber€€€€€€€€€
*
height К
height€€€€€€€€€
2

life_style$К!

life_style€€€€€€€€€
:
marital_status(К%
marital_status€€€€€€€€€
4
meal_type_y%К"
meal_type_y€€€€€€€€€
.
next_BMI"К
next_BMI€€€€€€€€€
:
nutrition_goal(К%
nutrition_goal€€€€€€€€€
P
place_of_meal_consumption3К0
place_of_meal_consumption€€€€€€€€€
(
priceК
price€€€€€€€€€
N
projected_daily_calories2К/
projected_daily_calories€€€€€€€€€
,
protein!К
protein€€€€€€€€€
f
$social_situation_of_meal_consumption>К;
$social_situation_of_meal_consumption€€€€€€€€€
(
tasteК
taste€€€€€€€€€
N
time_of_meal_consumption2К/
time_of_meal_consumption€€€€€€€€€
*
weight К
weight€€€€€€€€€
p 

 
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ Ґ
*__inference_model_12_layer_call_fn_7096254у}КЋЗћДЌБќ~ѕ{–x—u“r”o‘l’i÷f„лЎиўеё„–…¬їі≠¶ЯШС √ЉЏўџмнфхВГЩЪЧШ°Ґ©™±≤єЇќҐ 
¬ҐЊ
≥™ѓ
$
BMIК
BMI€€€€€€€€€
0
	age_range#К 
	age_range€€€€€€€€€
0
	allergens#К 
	allergens€€€€€€€€€
,
allergy!К
allergy€€€€€€€€€
.
calories"К
calories€€€€€€€€€
8
carbohydrates'К$
carbohydrates€€€€€€€€€
<
clinical_gender)К&
clinical_gender€€€€€€€€€
<
cultural_factor)К&
cultural_factor€€€€€€€€€
F
cultural_restriction.К+
cultural_restriction€€€€€€€€€
J
current_daily_calories0К-
current_daily_calories€€€€€€€€€
J
current_working_status0К-
current_working_status€€€€€€€€€
2

day_number$К!

day_number€€€€€€€€€
3

embeddings%К"

embeddings€€€€€€€€€А
0
	ethnicity#К 
	ethnicity€€€€€€€€€
$
fatК
fat€€€€€€€€€
(
fiberК
fiber€€€€€€€€€
*
height К
height€€€€€€€€€
2

life_style$К!

life_style€€€€€€€€€
:
marital_status(К%
marital_status€€€€€€€€€
4
meal_type_y%К"
meal_type_y€€€€€€€€€
.
next_BMI"К
next_BMI€€€€€€€€€
:
nutrition_goal(К%
nutrition_goal€€€€€€€€€
P
place_of_meal_consumption3К0
place_of_meal_consumption€€€€€€€€€
(
priceК
price€€€€€€€€€
N
projected_daily_calories2К/
projected_daily_calories€€€€€€€€€
,
protein!К
protein€€€€€€€€€
f
$social_situation_of_meal_consumption>К;
$social_situation_of_meal_consumption€€€€€€€€€
(
tasteК
taste€€€€€€€€€
N
time_of_meal_consumption2К/
time_of_meal_consumption€€€€€€€€€
*
weight К
weight€€€€€€€€€
p

 
™ "!К
unknown€€€€€€€€€Ґ
*__inference_model_12_layer_call_fn_7096422у}КЋЗћДЌБќ~ѕ{–x—u“r”o‘l’i÷f„лЎиўеё„–…¬їі≠¶ЯШС √ЉЏўџмнфхВГЪЧЩШ°Ґ©™±≤єЇќҐ 
¬ҐЊ
≥™ѓ
$
BMIК
BMI€€€€€€€€€
0
	age_range#К 
	age_range€€€€€€€€€
0
	allergens#К 
	allergens€€€€€€€€€
,
allergy!К
allergy€€€€€€€€€
.
calories"К
calories€€€€€€€€€
8
carbohydrates'К$
carbohydrates€€€€€€€€€
<
clinical_gender)К&
clinical_gender€€€€€€€€€
<
cultural_factor)К&
cultural_factor€€€€€€€€€
F
cultural_restriction.К+
cultural_restriction€€€€€€€€€
J
current_daily_calories0К-
current_daily_calories€€€€€€€€€
J
current_working_status0К-
current_working_status€€€€€€€€€
2

day_number$К!

day_number€€€€€€€€€
3

embeddings%К"

embeddings€€€€€€€€€А
0
	ethnicity#К 
	ethnicity€€€€€€€€€
$
fatК
fat€€€€€€€€€
(
fiberК
fiber€€€€€€€€€
*
height К
height€€€€€€€€€
2

life_style$К!

life_style€€€€€€€€€
:
marital_status(К%
marital_status€€€€€€€€€
4
meal_type_y%К"
meal_type_y€€€€€€€€€
.
next_BMI"К
next_BMI€€€€€€€€€
:
nutrition_goal(К%
nutrition_goal€€€€€€€€€
P
place_of_meal_consumption3К0
place_of_meal_consumption€€€€€€€€€
(
priceК
price€€€€€€€€€
N
projected_daily_calories2К/
projected_daily_calories€€€€€€€€€
,
protein!К
protein€€€€€€€€€
f
$social_situation_of_meal_consumption>К;
$social_situation_of_meal_consumption€€€€€€€€€
(
tasteК
taste€€€€€€€€€
N
time_of_meal_consumption2К/
time_of_meal_consumption€€€€€€€€€
*
weight К
weight€€€€€€€€€
p 

 
™ "!К
unknown€€€€€€€€€Ѓ
E__inference_output_0_layer_call_and_return_conditional_losses_7097904eєЇ/Ґ,
%Ґ"
 К
inputs€€€€€€€€€ 
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ И
*__inference_output_0_layer_call_fn_7097889ZєЇ/Ґ,
%Ґ"
 К
inputs€€€€€€€€€ 
™ "!К
unknown€€€€€€€€€†
%__inference_signature_wrapper_7096920ц}КЋЗћДЌБќ~ѕ{–x—u“r”o‘l’i÷f„лЎиўеё„–…¬їі≠¶ЯШС √ЉЏўџмнфхВГЪЧЩШ°Ґ©™±≤єЇњҐї
Ґ 
≥™ѓ
$
BMIК
bmi€€€€€€€€€
0
	age_range#К 
	age_range€€€€€€€€€
0
	allergens#К 
	allergens€€€€€€€€€
,
allergy!К
allergy€€€€€€€€€
.
calories"К
calories€€€€€€€€€
8
carbohydrates'К$
carbohydrates€€€€€€€€€
<
clinical_gender)К&
clinical_gender€€€€€€€€€
<
cultural_factor)К&
cultural_factor€€€€€€€€€
F
cultural_restriction.К+
cultural_restriction€€€€€€€€€
J
current_daily_calories0К-
current_daily_calories€€€€€€€€€
J
current_working_status0К-
current_working_status€€€€€€€€€
2

day_number$К!

day_number€€€€€€€€€
3

embeddings%К"

embeddings€€€€€€€€€А
0
	ethnicity#К 
	ethnicity€€€€€€€€€
$
fatК
fat€€€€€€€€€
(
fiberК
fiber€€€€€€€€€
*
height К
height€€€€€€€€€
2

life_style$К!

life_style€€€€€€€€€
:
marital_status(К%
marital_status€€€€€€€€€
4
meal_type_y%К"
meal_type_y€€€€€€€€€
.
next_BMI"К
next_bmi€€€€€€€€€
:
nutrition_goal(К%
nutrition_goal€€€€€€€€€
P
place_of_meal_consumption3К0
place_of_meal_consumption€€€€€€€€€
(
priceК
price€€€€€€€€€
N
projected_daily_calories2К/
projected_daily_calories€€€€€€€€€
,
protein!К
protein€€€€€€€€€
f
$social_situation_of_meal_consumption>К;
$social_situation_of_meal_consumption€€€€€€€€€
(
tasteК
taste€€€€€€€€€
N
time_of_meal_consumption2К/
time_of_meal_consumption€€€€€€€€€
*
weight К
weight€€€€€€€€€"3™0
.
output_0"К
output_0€€€€€€€€€µ
K__inference_user_embedding_layer_call_and_return_conditional_losses_7097614fмн/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "-Ґ*
#К 
tensor_0€€€€€€€€€ђ
Ъ П
0__inference_user_embedding_layer_call_fn_7097600[мн/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ ""К
unknown€€€€€€€€€ђ