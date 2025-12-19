import rule.character
import rule.gain_matrix
import isopower.calc_a
from isopower.optimal_triangle import OptimalTriangleFinder


def main():

    # ↑, →, 左下で3通り、ほぼ↓
    c1=rule.character.Character(0.50, rule.character.MatchupVector(0.4,0))
    c2=rule.character.Character(0.50, rule.character.MatchupVector(0,0.6))
    c3=rule.character.Character(0.50, rule.character.MatchupVector(-0.2,-0.2))
    c5=rule.character.Character(0.40, rule.character.MatchupVector(0.3,-0.2))
    c6=rule.character.Character(0.35, rule.character.MatchupVector(0,-1.0))
    environment=rule.gain_matrix.Pool([c1,c2,c3,c5,c6])
    A=environment.get_matrix()
    print("A")
    print(A)
    print(environment.get_pxy_list())

    finder = OptimalTriangleFinder(pool=environment)
    finder.find() 
    result=finder.get_result()
    print("By finder, a:",result[0])

    new_env = environment.convert([0.2,0.3])
    B=new_env.get_matrix()
    print("B")
    print(B)
    print(new_env.get_pxy_list())

def equilateral_triangle():
    ROOT3=1.7320508075688772
    c1=rule.character.Character(25, rule.character.MatchupVector( 2, 0))
    c2=rule.character.Character(10, rule.character.MatchupVector(-1, 1.732050))
    c3=rule.character.Character( 0, rule.character.MatchupVector(-1,-1.732050))
    calc=isopower.calc_a.aCalculator(c1,c2,c3)
    a=calc.calc()
    print("a:",a)

def B2A():

    # ↑, →, 左下で3通り、ほぼ↓
    c1=rule.character.Character(0.62, rule.character.MatchupVector(0.2,-0.3))
    c2=rule.character.Character(0.38, rule.character.MatchupVector(-0.2,0.3))
    c3=rule.character.Character(0.48, rule.character.MatchupVector(-0.4,-0.5))
    c5=rule.character.Character(0.53, rule.character.MatchupVector(0.1,-0.5))
    c6=rule.character.Character(0.55, rule.character.MatchupVector(-0.2,-1.3))
    environment=rule.gain_matrix.Pool([c1,c2,c3,c5,c6])
    B=environment.get_matrix()
    print("B")
    print(B)
    print(environment.get_pxy_list())

    finder = OptimalTriangleFinder(pool=environment)
    finder.find() 
    result=finder.get_result()
    print("By finder, a:",result[0])

    new_env = environment.convert(finder.get_a())
    A=new_env.get_matrix()
    print("A")
    print(A)
    print(new_env.get_pxy_list())


def squere_example():
    from example import squere
    A=squere.get_matrix()
    print(A)


if __name__ == "__main__":
    #main()
    #equilateral_triangle()
    #B2A()
    squere_example()