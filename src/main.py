import rule.character
import rule.gain_matrix


def main():
    c1=rule.character.Character(10, rule.character.MatchupVector(2,0))
    c2=rule.character.Character(10, rule.character.MatchupVector(-1,1.732050))
    c3=rule.character.Character(10, rule.character.MatchupVector(-1,-1.732050))
    environment=rule.gain_matrix.Environment([c1,c2,c3])
    A=environment.get_matrix()
    return A


if __name__ == "__main__":
    print(main())
