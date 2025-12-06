import rule.character
import rule.gain_matrix


def main():
    c1=rule.character.Character(0, rule.character.MatchupVector(2,0))
    c2=rule.character.Character(0, rule.character.MatchupVector(-1,1.732050))
    c3=rule.character.Character(0, rule.character.MatchupVector(-1,-1.732050))
    environment=rule.gain_matrix.Environment([c1,c2,c3])
    A=environment.get_matrix()
    print("A")
    print(A)
    print(environment.get_param())

    new_env = environment.convert([0.5,0.5])
    B=new_env.get_matrix()
    print("B")
    print(B)
    print(new_env.get_param())



if __name__ == "__main__":
    main()