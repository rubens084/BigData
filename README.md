# Bigdata
<br><strong>Unidad1</strong>
<p align="center">
<br><strong>Tecnológico Nacional de México</strong>
<br><strong>Instituto Tecnológico de Tijuana</strong>
<br><strong>Subdirección académica</strong>
<br><strong>Departamento de Sistemas y Computación</strong>
<br><strong>Semestre: ENERO - JUNIO 2020</strong>
<br><strong>Ingeniería en Tecnologías de la Información y Comunicaciones</strong>
<br><strong>Ingeniería Informatica</strong>
<br><strong>Materia: Datos Masivos</strong>
<br><strong>Unidad: 1</strong>
<br><strong>Dorado Aguilus Ruben #15210328</strong>
   <br><strong>Mejia Manriquez Rocio #14212336</strong>
<br><strong>Docente: Dr. Jose Christian Romero Hernandez</strong>
</p>

<h1>Practices</h1>
<h4>Instructions</h4>
<h4>#1</h4>

<li>Develop an algorithm in scala that calculates the radius of a circle.
<li> simple operations, where the radius of a 15cm circle is obtained, with the formula diameter / 2 * pi
print("Circulo de 15cm ")     
15/(2*3.1416)
<li>Develop an algorithm in scala that tells me if a number is a prime number.
<li>Given the variable bird = "tweet", use string interpolation to print "I am writing a tweet".
   val bird = "Tweet"
val greet = s"Estoy escribiendo en $bird"
  <li>play with variables that contains a word where you add it in which phrase
<li>Given the variable message = "Hola Luke soy tu padre!" use slilce to extract the sequence "Luke".
  <li>using the st slice you can cut part of a phrase or a word giving the parameters
 val st = "Hola Luke yo soy tu padre!"
st slice  (5,8)
<li> What is the difference of a value and a variable in scala ?.
   Variables in scala
 Values ​​(val) are immutable, once assigned cannot be changed.
 Variables (var) can be reallocated.
 Note, when reassigning you must use the same type of data!
<li>Given the tuple (2,4,5,1,2,3,3.1416,23) return the number 3.1416.
  <li>in a number array using._ you can extract a particular number
  val my_tup = (2,4,5,1,2,3,3.1416,23)
my_tup._7


<h4>Instructions</h4>
<h4>#2</h4>
<li>1. Create a list called "list" with the elements "red", "white", "black"
  <li> a list is created with val and the name by entering the data in this list
  <li> val lista = List("rojo","blanco","negro")
    <li> 2. Add 5 more items to "list" "green", "yellow", "blue", "orange", "pearl"
  <li> to add more data to the list, just add at the end of it the data you want to enter
<li> val lista = List("rojo","blanco","negro","verde","amarillo","azul","naranja","perla")
    <li> 3. Bring the "list" "green", "yellow", "blue" items
<li> we use slice again to take part in the list
    <li>lista slice (3,6)
  <li>4. Create a number array in the 1-1000 range in 5-in-5 steps
   <li>creating an arrangement is defined as fix.range and where it starts and where it 
     <li>ends and how much the jumps will be
   <li>Array.range(1,1000, 5)
   <li>5. What are the unique elements of the List list (1,3,3,4,6,7,3,7) use conversion to sets
  <li>var list2=List(1,3,3,4,6,7,3,7)
<li>list2.toSet
<li>6. Create a mutable map called names containing the following
  <li>"Jose", 20, "Luis", 24, "Ana", 23, "Susana", "27"
 <li>val names = collection.mutable.Map(("Jose", 20), ("Luis", 24), ("Ana", 23),("Susana",27))
  <li> 6 a. Print all map keys
    <li>names.keys
    <li>  b. Add the following value to the map ("Miguel", 23)
      <li> names += ("Miguel" -> 23)
        
    
<li>every number after the first two is the sum of the two preceding numbers.
<li>To simplify: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144,
<li>The number of times the function is called causes a stack overflow in most languages.

<li>Algorithm Downward recursive version....
<li> If the number enterd is less than 2, the number will be returnd
<li> if is less than 2, the function will do a series of operations, and  returns the result  
def  function(num: Int): Int =  {  
if (num<2)  {  
return num  }  
else  {  
return function(num-1) + function(num-2)  }  
}   
function(10) 
///////////////////////////////////////////////////
<li>Version with explicit formula 
<li>If the number enterd is less than 2, the number will be returnd
<li> if is not less than 2, the proses of the formula is divided into parts to finally create the result.
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

<li>In this algorithm a function was added that after having performed the corresponding 
<li>operations the function will give us a result (return) this must be an integer value (Int)
<li> a cycle (for) starts where k = 1, will start cycling until it becomes (num)
<li>(num) represents the value that will be entered into the function
<li> Depending on the cycle (for) the variables (x, y, z) will begin to change
<li> your result until the end of the cycle (for) The result will be returned with (return)
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
<li>In this a function was added after having performed the corresponding operations the function
<li> will give us a result. this must be an integer value (Int) A cycle (for) starts where k = 1,
<li> will start cycling until it becomes (num) this one represents the value that will be enterd 
<li> into the function Depending on the cycle (for) the variables (x, y) will begin to change their values
<li>until the end of the cycle (for).
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
<li> In this algorithm a function asks for an value (Int) then return an integer value with decimals
<li> an array is created that starts from 0 to (num + 1) if the variable (num) is less than 2, 
<li> that same variable is returned as a result, Otherwise the vector with space (0) will have a 
<li> value of (0) and the vector with space (1) will have a value of (1) Start cycling with a for
<li>the vector The result will be the variable (num).
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
<li> In this algorithm a function was added that after having performed the corresponding 
<li>operations the function will give us a result (return) this must be an integer value with 
<li> decimal(Double). If the value entered is less than or equal to 0, then that value 
<li> will be returned Otherwise you will have to make a series of operations Start a cycle 
<li> (while) where the variables will begin to change value depending on the cycle iteration
<li> If variable (i) is odd, different operations will be done If variable (i) is even, different 
<li>operations will be done Variable (i) will begin to change value each time the cycle is entered
<li>until you exit the cycle, and the sum of (a + b) is returned

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


<li>Pearson's Correlation


<li>In statistics, Pearson's correlation coefficient is a linear measure between
<li>two quantitative random variables. Unlike covariance, Pearson's correlation
<li>is independent of the scale of measurement of the variables.
<li>A Pearson correlation is a number between -1 and 1 that indicates the extent
 <li>to which two variables are linearly related. The Pearson correlation is also 
 <li>known as the “product moment correlation coefficient” (PMCC) or simply “correlation”.
 <li>
 <li>
 <li><li>.(a) ----------------------------------------------.(b)
     <li<li>>-
       <li><li>-
          <li>-
              <li><li>-
                 <li><li> -
                      <li>-   
                     <li> <li>  -   
                         <li><li>    .(c)

<li><li>it is used to group data, in this case we have 3 points a, b, c where group a is a type
<li>color like red and b is another type like green to which color does c belong if we 
<li>go in the distance at a smaller distance is the one that belongs
   
   
 <li>  import org.apache.spark.sql.SparkSession

<li> val spark = SparkSession.builder().getOrCreate()

<li> val df = spark.read.option("header", "true").option("inferSchema","true")csv("Sales.csv")

<li> //Company, Person, Sales


 <li> //Group By on single column
 <li> df.groupBy("Person").count().show(false)
  <li>df.groupBy("Person").avg("Sales").show(false)
  <li>df.groupBy("Person").sum("Sales").show(false)
 <li> df.groupBy("Person").min("Sales").show(false)
 <li> df.groupBy("Person").max("Sales").show(false)
 <li> df.groupBy("Person").mean("Sales").show(false)

  <li>//GroupBy on multiple columns
 <li> df.groupBy("Company", "Person")
   <li> .sum("Sales","Sales")
  <li>  .show(false)
 <li> df.groupBy("department","state")
   <li> .avg("salary","bonus")
   <li> .show(false)
 <li> df.groupBy("department","state")
  <li>  .max("salary","bonus")
   <li> .show(false)
 <li> df.groupBy("department","state")
   <li> .min("salary","bonus")
   <li> .show(false)
 <li> df.groupBy("department","state")
  <li>  .mean("salary","bonus")
   <li> .show(false)

  <li>  //Running Filter
 <li> df.groupBy("department","state")
  <li>  .sum("salary","bonus")
  <li>  .show(false)

  <li>   //using agg function
 <li> df.groupBy("Company")
  <li>  .agg(
    <li>  sum("Person").as("sum_salary"),
    <li>  avg("Person").as("avg_salary"),
    <li>  sum("Sales").as("sum_bonus"),
   <li>   max("Sales").as("max_bonus"))
 <li>   .show(false)

<li> df.groupBy("Company")
   <li> .agg( sum("Person").as("sum_salary")).show(false)


<li>  df.groupBy("department")
  <li>  .agg(
     <li> sum("salary").as("sum_salary"),
     <li> avg("salary").as("avg_salary"),
     <li> sum("bonus").as("sum_bonus"),
     <li> stddev("bonus").as("stddev_bonus"))
   <li> .where(col("sum_bonus") > 50000)
  <li>  .show(false)
<li> }
   
