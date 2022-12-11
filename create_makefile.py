
Да да, я настолько ленивый что даже не сделал каталог картинок и не написал к нему makefile

images/problem_set_01/

img01_regr, img01_dobronet,
img02_dobronet, img02_dobronet_sol,
img03_net,
img04_perp1, img04_perp2, img04_perp3, img04_flat, img04_sol1, img04_sol2, img04_sol3, img04_sol4
img05_log_table, img05_sol1, img05_sol2, img05_sol3
img06_xor,
img08_oboi, img08_triangle, img08_krest, img08_hard, 
img09_one_step, img09_net, 


images/problem_set_03/

img01_gr1, img01_gr2
img02_task, img02_sol1, img02_sol2
img05_task, img05_forpass, img05_backpass



xelatex img02_task.tex

# https://formulae.brew.sh/formula/imagemagick
# brew install imagemagick

convert -density 600 -flatten -resize 50% img01.pdf img01.png
convert -density 600  img01.pdf img01.png


xelatex img01_gr1.tex 
convert -density 600  img01_gr1.pdf img01_gr1.png

rm -f *.log && rm -f *.aux
