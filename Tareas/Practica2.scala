//Practice 2
// 1. Crea una lista llamad "lista" con los elementos "rojo", "blanco", "negro"

val lista = List("rojo","blanco","negro")
print("///////////////////////////////////////////////////////")
// 2. Añadir 5 elementos mas a "lista" "verde" ,"amarillo", "azul", "naranja", "perla"
val lista = List("rojo","blanco","negro","verde","amarillo","azul","naranja","perla")
print("///////////////////////////////////////////////////////")
// 3. Traer los elementos de "lista" "verde", "amarillo", "azul"
lista slice (3,6)
print("///////////////////////////////////////////////////////")
// 4. Crea un arreglo de numero en rango del 1-1000 en pasos de 5 en 5
Array.range(1,1000, 5)
print("///////////////////////////////////////////////////////")
// 5. Cuales son los elementos unicos de la lista Lista(1,3,3,4,6,7,3,7) utilice conversion a conjuntos
val s = conjuntos (1,3,3,4,6,7,3,7)
lista2.soSet
print("///////////////////////////////////////////////////")
// 6. Crea una mapa mutable llamado nombres que contenga los siguiente
//     "Jose", 20, "Luis", 24, "Ana", 23, "Susana", "27"
val names = collection.mutable.Map(("Jose", 20), ("Luis", 24), ("Ana", 23),("Susana",27))
print("///////////////////////////////////////////////////")
// 6 a . Imprime todas la llaves del mapa
names.keys
// 7 b . Agrega el siguiente valor al mapa("Miguel", 23) 
names += ("Miguel" -> 23)
