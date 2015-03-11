--****************************MACHINE LEARNING 2015 PRACTICAL ASSIGNMENT 6*************************--
--***********************************TRINITY TERM 2015*******WEEK 8********************************--

--terminal output: 
ion@ion-VirtualBox:timedatectl & th train.lua -vocabfile vocab.t7 -datafile train.t7 &
ion@ion-VirtualBox:/media/uDev/machinelearning/prac6$      
  Local time: Wed 2015-03-11 13:20:11 GMT
  Universal time: Wed 2015-03-11 13:20:11 UTC
        Timezone: Europe/London (GMT, +0000)
     NTP enabled: yes
NTP synchronized: no
 RTC in local TZ: no
      DST active: no
 Last DST change: DST ended at
                  Sun 2014-10-26 01:59:59 BST
                  Sun 2014-10-26 01:00:00 GMT
 Next DST change: DST begins (the clock jumps one hour forward) at
                  Sun 2015-03-29 00:59:59 GMT
                  Sun 2015-03-29 02:00:00 BST
-- ignore option print_every	
-- ignore option save_every	
-- ignore option savefile	
-- ignore option vocabfile	
-- ignore option datafile	
loading data files...	
cutting off end of data so that the batches/sequences divide evenly	
reshaping tensor...	
data load done.	
cloning criterion	
cloning lstm	
cloning softmax	
cloning embed	

iteration   10, loss = 2.65157351, gradnorm = 6.1938e+00	--training makes luajit eat up
iteration  100, loss = 2.16414752, gradnorm = 3.8787e+00	--90% of vm's allocated 
iteration  200, loss = 2.05828568, gradnorm = 4.3131e+00    --processing power & 1.8G of RAM
iteration  300, loss = 2.28432427, gradnorm = 4.3981e+00	[...]
iteration 7120, loss = 1.52723811, gradnorm = 5.1372e+00	--training stopped here.

--samples after 7120 iterations of sgd for training:
president, tredis. im, genarily be announchels, the head which an earching. when parked to be 
were salism med flimhrieldefire quice stald, cannertoz likee felt hemsee however eministt got 
part, is glo

--samples after 100 iterations of sgd for training: 
oof dornopeswerine heppenc.. proin angeions, the infeh dexsrat alance funtrith liof frists
ands lent ition in forkest kiranted mol. to canmacit outu miken edmicents bletrored che. shy
froan stales fos 

--samples after 600 iterations of sgd for training:
rightergilemt siman flten and social silies still flight with i jeffed theire new becially
andeling and of ne geminital. hed form adque cond muchort jremectder gisted hown he dedaninn
hen part fire. t