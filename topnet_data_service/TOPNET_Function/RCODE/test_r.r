#install.packages("XML")
library(XML)
library(Hmisc)
library(soilDB)
library(SSOAP)
library(RCurl)




q1 <-  "SELECT component.mukey, component.cokey, compname, comppct_r, hzdept_r, hzdepb_r, hzname,awc_r, ksat_r,wthirdbar_r,wfifteenbar_r,dbthirdbar_r,\n            sandtotal_r,claytotal_r,om_r\n            FROM component JOIN chorizon ON component.cokey = chorizon.cokey\n            AND mukey IN ('154386','154456','154457','154459','154460','154462','154464','154509','482654','482656','482658','482666','482667','482668','482669','482670','482671','482677','482678','482681','482682','482691','482692','482694','482695','482696','482703','482711','482713','482714','482715','482716','482718','482719','482720','482721','482722','482723','482724','482726','482728','482735','482736','482737','482738','482740','482742','482743','482744','482745','482750','482752','482755','482759','482760','482761','482762','482765','482770','482772','482773','482775','482776','482778','482779','482780','482781','482783','482784','482785','482789','482790','482791','482792','482797','482798','482799','482800','482801','482802','482806','482808','482811','482812','482813','482814','482815','482816','482817','482818','482819','482821','482824','482835','482838','482839','482840','482841','482842','482843','482844','482845','482846','482847','482848','482849','482850','482854','482855','482857','482858','482861','482864','485187','485191','485194','485197','485198','485200','485203','485204','485206','485207','485208','485209','485217','485218','485219','485222','485223','485224','485225','485226','485227','485232','485234','485236','485238','485241','485243','485245','485247','485248','485249','485250','485254','485255','485256','485258','485261','485262','485263','485273','485276','485278','485280','485284','485285','485289','485290','485295','485296','485297','485298','485300','485301','485306','485309','485311','485314','485315','485316','485318','485319','485321','485322','485324','485325','485331','485333','485335','485336','485341','485343','485349','503799','503800','503804','503805','503819','503820','503828','503834','503847','503850','503854','503855','503858','503859','503860','503870','503871','503902','503910','503922','503927','503928','632021','632059','780666','789457','789540','995835','1387629','1487320','2396255','2396259','2511840','2822436')ORDER BY mukey, comppct_r DESC, hzdept_r ASC"

print (q1)
# now get component and horizon-level data for these map unit keys
res <- SDA_query(q1)

print (res)