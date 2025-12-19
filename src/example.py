import rule.character
import rule.gain_matrix
import rule.score
import numpy as np

c1=rule.character.Character(0.50, rule.character.MatchupVector(0.4,0))
c2=rule.character.Character(0.50, rule.character.MatchupVector(0,0.6))
c3=rule.character.Character(0.50, rule.character.MatchupVector(-0.2,-0.2))
c5=rule.character.Character(0.40, rule.character.MatchupVector(0.3,-0.2))
c6=rule.character.Character(0.35, rule.character.MatchupVector(0,-1.0))
main_pool=rule.gain_matrix.Pool([c1,c2,c3,c5,c6])

squereC=rule.character.Character(0.50, rule.character.MatchupVector(-0.3,0.1))
squereD=rule.character.Character(0.50, rule.character.MatchupVector(-0.1,-0.3))
squere=rule.gain_matrix.Pool([c1,c2,squereC,squereD])


def main(pool:rule.gain_matrix.Pool):
    A=pool.get_matrix()

    nash_v=np.array([0.272727,0.181818,0.545454,0,0])

    score = rule.score.calc_score(A,nash_v,nash_v)
    print(f"score:{score}")

    w = rule.score.calc_weight(A,nash_v)
    print(f"w:{w}")


if __name__ =="__main__":
    main(main_pool)