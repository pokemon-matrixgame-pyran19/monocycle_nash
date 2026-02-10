# 2×2利得行列のゲーム
2×2の利得行列Aがある。
$$
A=
\begin{pmatrix}
a c
b d
\end{pmatrix}
$$

この時ゲームの値はgは次で与えられる。

$$
g=(ad-bc)/(a+d-b-c)
$$


## 2×2ゲームかつ単相性モデル

特にa,b,c,dが次のように書ける時を考える.(vは二次元ベクトルでv1×v2=v1x v2y - v1y-v2x)

a=p1-p3 +v1×v3
b=p2-p3 +v2×v3
c=p1-p4 +v1×v4
d=p2-p4 +v2×v4
ついでに以下も定義
e=p1-p2 +v1×v2
f=p3-p4 +v3×v4


この時ゲームの値gは次
$$
g=(ef+M)/(v_1-v_2)\times(v_3-v_4)
$$
ただしMは次の4×4行列式
$$
\begin{vmatrix}
1&1&1&1\\
p1&p2&p3&p4\\
v1&v2&v3&v4
\end{vmatrix}
$$



