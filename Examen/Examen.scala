val arre= ((11, 2, 4), (4, 5, 6), (10, 8, -12))

def Sumadiagonales (arreglotuplas: ((Int, Int, Int), (Int, Int, Int), (Int, Int, Int))) : Int ={
    val diagonal_1 = (arre._1._1) + (arre._2._2) + (arre._3._3)
    val diagonal_2 = (arre._1._3) + (arre._2._2) + (arre._3._1)

    var diagonaldifference = diagonal_1 - diagonal_2
    var resultado = math.abs (diagonaldifference)
    return resultado
}

Sumadiagonales(arre)