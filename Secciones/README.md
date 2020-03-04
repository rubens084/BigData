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



<a href="https://github.com/rubens084/Bigdata/blob/Unidad1/Secciones/Seccion_1.scala">Practica 1</a>



<li>100 REPL res0: Int = 100
<li>2.5 res1: Double = 2.5
<li>1+1
<li>2-1
<li>2*5
<li>1/2 Int = 0
<li>1.0/2 Double = 0.5
<li>1/2.0
<li>1.0/2.0

<li>Exponentials
math.pow(4, 2) //Double = 16.0
res0 // Int = 100
res0 + res11 //Int = 200
1 + 2 * 3 + 4 //Int = 11
(1+2) * (3+4) //Int = 21

<li>fett to meters
<li>3 * 0.3048 //Double = 0.914400

<li> Variables in scala
<li> Values (val) are immutable, once they are assigned they
<li>can not be changed.
<li>Variables (var) can be ressigned.
<li>Note, when reassigning you must used the same data type!

<li>Data Types
<li>Int
<li>Double
<li>String
<li>Doolean
<li>Collections

var myvar:Int = 10
val myval: Double = 2.5

<li>val <name> : <type> = <literal>
<li>var <name> : <type> = <literal>

var myvar: Int = 10
val mival: Double = 2.5

myvar = "hello" //error: type mistmatch
myvar = 100
myvar = "hello" //error: reassignment to val
myval = 10.1 //error: reassigment to val

val c = 12 //Scala reassing Int 12
val my_string = "Hello" 

<li>Can not do this val 23my_string nor my.string

<li>Booleans and comparison operators
true
false
1 > 2 //false
1 < 2 //true
1<= 30 //true
2 == 2 //true
2 != 4 //true

4 % 2 //0mod operator
5 % 2 // 1
6 % 2 // if 0 the number is even

<li>string
println("Hello")
//   println('Hello') //Error unclosed charracter literal
val greeting = "Hello " + "there!"
"dance"*5
val st = "hello"
val name = "Cristian"
val greet = s"Hello ${name}"
val greet = s"Hello $name"

printf("A string %s, an integer %d, a float %f", "Hi",10,3.1416)
printf("A float %1.2f", 1.2345)
printf("A float %1.2f", 1.2395)

val st = "This a long string"
st.charAt(0)
st.charAt(3)
st.indexof("a")
st slice (0.4)
//Grab the word long of st


<a href="https://github.com/rubens084/Bigdata/blob/Unidad1/Secciones/Seccion_2.scala">Practica 2</a>
/////////////////////////////////////////
val st = "item1 item2 item3"
se matches "item1 item2 item3" //res0: Boolean = false
st matches "item2" //res1: Boolean = true
st contains "item1" // res2: true
st contains "item4" // res: false

<li>Tuples
(1, 2.3, "Hello") // res4: (Int, Double, String) = (1, 2.3, Hello)
val my_tup = (1,2.2 "hello,", 23.2, true)
<li> res4: (Int, Double, String, Double, Boolean) = (1,2.2,"Hello", 23.2, true)
(3,1,(2,3))
<li> res5: (Int, Int, (Int, Int)) = (3,1,(2,3))
my_tup._3 //res6: String = hello
my_tup._5 //res7: Boolean = true


<li> Collections Lists, arrays,
val evens = List(2,4,6,8,10) //evens: List(1, 2.2, true)

List(1,2.2,true) //res8: List[AnyVal] = List(1, 2.2, true)
evens(0) //res9: Int = 2
evens(4) //res10: Int = 10
evens.head //res11:Int = 2
evens.tail //res12: List[Int] = List(4, 6, 8, 10)
val my_list = List(List(1,2,3),List(3,2,1))
val my_list = List(("a",1), ("b",2), ("c",3))
val my_list = List(1,5,3,7,6,109)
my_list.lift
my_list.sorted
my_list.size
my_listmax
my_list.min
my_list.sum
my_list.product

val z = List(4,5,6,7)
z.drop(2)
z.takeRight(1)
z.takeRight(3)

val x = List(1,2,3,4,5,6,7,8)
x slice (0,3)
x slice (3, 6)

/////////////////////////////////////////////
<a https://github.com/rubens084/Bigdata/blob/Unidad1/Secciones/Seccion_3.scala">Practica 3</a>


@@ -44,4 +44,8 @@ my_list.product
val z = List(4,5,6,7)
z.drop(2)
z.takeRight(1)
z.takeRight(3) 
z.takeRight(3)

val x = List(1,2,3,4,5,6,7,8)
x slice (0,3)
x slice 3, 6)
 62  Scala_Basics/Session_03.scala 
@@ -0,0 +1,62 @@
<li> Arrays
<li> Arrays are mutable, List are not
val arr = Array(3,4,5)
val arr = Array("a","b","c")
val arr = Array("a","b", true, 1.2)

<li>Create arrays  with range method
Array.range(0, 10)
Array.range(0, 10, 2)

Range(0,5)

<li>Sets not cotains duplicate elements
val s = Set()
val s = Set(1,2,3)

val s = Set(2,2,2,3,3,3,5,5,5)

val s = collection.mutable.Set(1,2,3)
s += 4

val ims = collection.mutable.Set(2,3,4)
ims += 5
ims.add(6)
ims

ims.max
ims.min

val mylist = List(1,2,3,1,2,3)
mylist.toSet

val newset = mylist.toSet
newset

<li>Maps key value pair storage

val mymap = Map(("saludo", "Hola"), ("pi", 3.1416), ("z", 1.3))
mymap("pi")
mymap("saludo")
mymap("ja")
mymap get "pi"
mymap get "z"
mymap get "o"

val mutmap = collection.mutable.Map(("z", 123), ("a", 5), ("b", 7))

mutmap += ("lucky" -> 777)
mutmap
mutmap.keys
mutmap.values

////////////////////////////////////////////////////////////
<a https://github.com/rubens084/Bigdata/blob/Unidad1/Secciones/Seccion_4.scala">Practica 4</a>

<li>Control Flow
<li>if, else if, and else

<li> Genaral syntax ////
<li> if(boolean){
<li>   do something    
<li> }else if (boolean){
<li>  do something else   
<li> }else{
<li>  do somethin if none of the booleans were true     
<li> }

if(true){
    println("If you are true I will print!!")
}

if(3 == 3){
    println("3 is iqual to 3")
}

val x = "hello"

if(x.endsWith("o")){
    println("The string ends with letter o!!")
}

val x = "hellox"

if(x.endsWith("o")){
    println("The string ends with letter o!!")
}else{
    println("The string does not end with o!!")
}

val person = "Christian"

if(person == "Jose"){
    println("Hello Jose!")
}else if (person == "Christian"){
    println("Welcome to scala basics Christian!")
}else{
    println("Hello anonymous person!!")
}

<li>AND
println((1 == 2) && (3 == 3))
<li>OR
println((1 == 2) || (3 == 3))
<li>NOT
println(!(1 == 1))


<li>For Loopps

<li> Genaral syntax  ////////////////////

<li> for(item <- interable_sequence){
<li>     do something
<li> }
for(i <- List("Hugo", "Paco", "Luis")){
    println("Hello " + i)
}

for(i <- Array.range(0,5)){
    println(i)
}

for(i <- Set(1,2,3)){
    println(i)
}

for(num <- Range(0,10)){
    if(num%2 == 0){
        println(s"$num is even")
    }else{
        println(s"$num is odd")
    } 
}

val names = List("Juan", "Luis", "Hugo", "Chrtistian", "Carlos")

for(name <- names){
    if(name.startsWith("C")){
        println(s"$name starts with a C")
    }
}
////////////////////////////////////////////////////////////////
<a https://github.com/rubens084/Bigdata/blob/Unidad1/Secciones/Seccion_5.scala">Practica 5</a>

<li>While loop 

import util.control.Breaks._

var x = 0

while(x < 5){
    println(s"x is currently $x")
    println(s"x is still less then 5, adding 1 to x ")
    x = x + 1
}

var y = 0

while(y < 10){
    println(s"y is currently $y")
    println(s"y is still less then 10, add 1 to y")
    y = y+1
    if(y==3) break 

}
println("###########")

<li>Functions 

def simple(): Unit = {
    println("simple print")
}

simple()

def adder(num1:Int, num2:Int): Int = {
    return num1 + num2
}

adder(5, 5)

def greetName(name:String): String={
    return s"Hello $name"
}

val fullgreet = greetName("Christian")
println(fullgreet)

def isPrime(num:Int): Boolean = {
    for(n <- Range(2, num)){
        if(num%n == 0){
            return false
        }
    }
    return true
}

println(isPrime(10))
println(isPrime(23))

val numbers = List(1,2,3,7)

def check(nums:List[Int]): List[Int]={
    return nums
}

println(check(numbers))

<a https://github.com/rubens084/Bigdata/blob/Unidad1/Secciones/Seccion_6.scala">Practica 6</a>

 
<li> def isEven(num:Int): Boolean = {
<li>     return num%2 == 0
<li> }
<li> def isEven(num:Int): num%2 == 0
<li> println(isEven(6))
<li> println(isEven(3))

def listEvens(list:List[Int]): String ={
    for(n <- list){
        if(n%2==0){
            println(s"$n is even")
        }else{
            println(s"$n is odd")
        }
    }
    return "Done"
}

val l = List(1,2,3,4,5,6,7,8)
val l2 = List(4,3,22,55,7,8)
listEvens(l)
listEvens(l2)

<li>3 7 afortunado

def afortunado(list:List[Int]): Int={
    var res=0
    for(n <- list){
        if(n==7){
            res = res + 14
        }else{
            res = res + n
        }
    }
    return res
}

val af= List(1,7,7)
println(afortunado(af))

def balance(list:List[Int]): Boolean={
    var primera = 0
    var segunda = 0

    segunda = list.sum

    for(i <- Range(0,list.length)){
        primera = primera + list(i)
        segunda = segunda - list(i)

        if(primera == segunda){
            return true
        }
    }
    return false 
}

val bl = List(3,2,1)
val bl2 = List(2,3,3,2)
val bl3 = List(10,30,90)

balance(bl)
balance(bl2)
balance(bl3)

def palindromo(palabra:String):Boolean ={
    return (palabra == palabra.reverse)
}

val palabra = "OSO"
val palabra2 = "ANNA"
val palabra3 = "JUAN"

println(palindromo(palabra))
println(palindromo(palabra2))
println(palindromo(palabra3))
