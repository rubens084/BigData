//The Fibonacci sequence is, by definition, the integer sequence in which 
//every number after the first two is the sum of the two preceding numbers.
// To simplify: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144,
// The number of times the function is called causes a stack overflow in most languages.

//Algorithm Downward recursive version....
// If the number enterd is less than 2, the number will be returnd
// if is less than 2, the function will do a series of operations, and  returns the result  
def  function(num: Int): Int =  {  
if (num<2)  {  
return num  }  
else  {  
return function(num-1) + function(num-2)  }  
}   
function(10) 
///////////////////////////////////////////////////
//Version with explicit formula 
// If the number enterd is less than 2, the number will be returnd
//  if is not less than 2, the proses of the formula is divided into parts to finally create the result.
def  function1(num: Double): Double =  {   
if (num<2)  {
return num}  
else  {  
var w = ((1+(Math.sqrt(5)))/2)  
var x = Math.pow(w,num)  
var y = Math.pow((1-w),num)  
var z = ((x-(y)))/(Math.sqrt(5))  
return(z)  }  }  
function1(10)
///////////////////////////////////////////////////////

// In this algorithm a function was added that after having performed the corresponding 
// operations the function will give us a result (return) this must be an integer value (Int)
// a cycle (for) starts where k = 1, will start cycling until it becomes (num)
// (num) represents the value that will be entered into the function
// Depending on the cycle (for) the variables (x, y, z) will begin to change
// your result until the end of the cycle (for) The result will be returned with (return)
def function2 (num: Int): Int = {
var x = 0
var y = 1
var z = 0
for (k <- 1 to num)
{
z = y + x
x = y
y = z
}
 return (x)
}
function2 (10)
///////////////////////////////////////////////////////////////////
// In this a function was added after having performed the corresponding operations the function
// will give us a result. this must be an integer value (Int) A cycle (for) starts where k = 1,
// will start cycling until it becomes (num) this one represents the value that will be enterd 
// into the function Depending on the cycle (for) the variables (x, y) will begin to change their values
// until the end of the cycle (for).
def function3 (num: Int): Int = {
var x = 0
var y = 1
for (k <- 1 to num)
{
y = y + x
x = y - x
}
return (x)
}
function3 (10)
////////////////////////////////////////////////////
// In this algorithm a function asks for an value (Int) then return an integer value with decimals
// an array is created that starts from 0 to (num + 1) if the variable (num) is less than 2, 
// that same variable is returned as a result, Otherwise the vector with space (0) will have a 
// value of (0) and the vector with space (1) will have a value of (1) Start cycling with a for
// the vector The result will be the variable (num).
def function4 (num: Int): Double =
{
val vector = Array.range (0, num + 1)
if (num <2)
{
return (num)
}
else
{
vector (0) = 0
vector (1) = 1
for (k <- 2 to num)
{
vector (k) = vector (k-1) + vector (k-2)
}
return vector (num)
}
}
function4 (10)

/////////////////////////////////////////////////
// In this algorithm a function was added that after having performed the corresponding 
// operations the function will give us a result (return) this must be an integer value with 
// decimal(Double). If the value entered is less than or equal to 0, then that value 
// will be returned Otherwise you will have to make a series of operations Start a cycle 
// (while) where the variables will begin to change value depending on the cycle iteration
// If variable (i) is odd, different operations will be done If variable (i) is even, different 
// operations will be done Variable (i) will begin to change value each time the cycle is entered
// until you exit the cycle, and the sum of (a + b) is returned

def function5 (n: Double): Double =
{
if (n <= 0)
{
return (n)
}
else
{
var i: Double = n - 1
var auxOne: Double = 0
var auxTwo: Double = 1
var a: Double = auxTwo
var b: Double = auxOne
var c: Double = auxOne
var d: Double = auxTwo
while (i> 0)
{
if (i% 2 == 1)
{
auxOne = (d * b) + (c * a)
auxTwo = ((d + (b * a)) + (c * b))
a = auxOne
b = auxTwo
}
else
{
var pow1 = Math.pow (c, 2)
var pow2 = Math.pow (d, 2)
auxOne = pow1 + pow2
auxTwo = (d * ((2 * (c)) + d))
c = auxOne
d = auxTwo
}
i = (i / 2)
}
return (a + b)
}
}
function5 (9)


