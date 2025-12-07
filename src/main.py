import rule.character
import rule.gain_matrix
import isopower.calc_a


def main():
    c1=rule.character.Character(0, rule.character.MatchupVector(2,0))
    c2=rule.character.Character(0, rule.character.MatchupVector(-1,1.732050))
    c3=rule.character.Character(0, rule.character.MatchupVector(-1,-1.732050))
    environment=rule.gain_matrix.Environment([c1,c2,c3])
    A=environment.get_matrix()
    print("A")
    print(A)
    print(environment.get_param())

    new_env = environment.convert([-2.88,6.66])
    B=new_env.get_matrix()
    print("B")
    print(B)
    print(new_env.get_param())

def equilateral_triangle():
    ROOT3=1.7320508075688772
    c1=rule.character.Character(25, rule.character.MatchupVector( 2, 0))
    c2=rule.character.Character(10, rule.character.MatchupVector(-1, 1.732050))
    c3=rule.character.Character( 0, rule.character.MatchupVector(-1,-1.732050))
    calc=isopower.calc_a.aCalculator(c1,c2,c3)
    a=calc.calc()
    print("a:",a)


if __name__ == "__main__":
    #main()
    equilateral_triangle()