# Authors: Nazmus Sazib, Prasanna Dahal, Dylan Beaudette

args<-commandArgs(TRUE)

# args = c('E:/TOPNET/test_folder/test2','Soil_mukey.tif',
#         'dth1.tif', 'dth2.tif', 'psif.tif', 'sd.tif',
#        'bbl.tif', 'psd.tif', 'rsm.tif', 'ssm.tif', 'ksat_rawls.tif', 'ksat_hz.tif')

require(raster)
require(plyr)
require(Hmisc)
require(soilDB)
require(SSOAP)
require(foreign)
require(shapefiles)
library(rgdal)

print ("1. All the packages loaded" )

dir=args[1]
setwd(dir)

r <- raster(args[2]) # get grid info from gssurgo data (download soil spatial and tabular data as tif format is big size)
r <- ratify(r,count=TRUE)
rat <- levels(r)[[1]]
mu=data.frame(rat)
names(mu)[1] <- 'MUKEY'
names(mu)[2] <- 'Count'
#mu=mu[-1,]
in.statement <- format_SQL_in_statement(mu$MUKEY)


##make query for each variable
print ("2. about to make a query" )
q1 <- paste("SELECT component.mukey, component.cokey, compname, comppct_r, hzdept_r, hzdepb_r, hzname,awc_r, ksat_r,wthirdbar_r,wfifteenbar_r,dbthirdbar_r,
            sandtotal_r,claytotal_r,om_r
            FROM component JOIN chorizon ON component.cokey = chorizon.cokey
            AND mukey IN ", in.statement, "ORDER BY mukey, comppct_r DESC, hzdept_r ASC", sep="")
print (q1)
# now get component and horizon-level data for these map unit keys
res <- SDA_query(q1)

print ("3. Querry successful" )
write.csv(res,'res.csv')


# ##query for soil depth. Almost always results NA values for most MUKEY. so no use...
# q2 <- paste("SELECT mukey,brockdepmin
#             FROM muaggatt
#             WHERE  mukey IN ", in.statement, "
#             ", sep="")
#
# res2 <- SDA_query(q2)


# # function for copmuting weighted-mean  within a component key unit
co.mean.whc <- function(i) {
  # Prasanna added the variable ksat_hz because he did not understand extra calc done to get the value for variable ksat

  # # # # # # # About how this function works (I think.. Prasanna)# # # # # # #
  # Because this is used in ddply, where c(mukey,cokey) is passed, i.e. two fields are passed,
  # so the fucntion automatically does the calculations for each unique combinations of mukey, and cokey.
  # For e.g., lets say for 1mukey, there exists 3 cokeys, and in each cokeys there exist 4horizons each.
  # The dataframe is split to many dataframe of uniqe mukey and cokey combinations. And all the horizons are
  # finally aggregated by the code below....
  # # # # # # #


  wt <- i$comppct_r[1]                                 # keep the first component pct (they are all the same)
  thick1 <- with(i, hzdepb_r - hzdept_r)               # compute horizon thickness
  thick=thick1/100 ##in m
  #ksat <- thick/i$ksat_r #compute saturated hydraulic conductivity
  #ksat<-i$ksat_r[1]*3600*10^(-6)

  ksat_hz <- thick * i$ksat_r                          # compute ksat
  wcthirdbar <- thick * i$wthirdbar_r                  # compute water content at 1/3 bar with horizon
  wctfifteendbar  <- thick * i$wfifteenbar_r           # compute water content at 15 bar with horizon
  dbthirdbar  <- thick * i$dbthirdbar_r                # compute density at 1/3 bar with horizon
  sand <- thick * i$sandtotal_r                        # compute percentage of sand by weight with horizon
  clay<- thick * i$claytotal_r                         # compute percentage of clay  by weight with horizon
  om <- thick * i$om_r                                 # compute percentage of organic matter by weight  with horizon
  awc <-  i$awc_r                                      # available water capacity

  thick.total=sum(thick, na.rm=TRUE)
  awc.total=sum(awc, na.rm=TRUE)
  awc.depth=( sum((thick *((!is.na(awc))*1)),na.rm=TRUE))
  #ksat.total <- sum(thick, na.rm=TRUE)/(sum(ksat, na.rm=TRUE)) # Harmonic mean
  #ksat.total<-ksat

  ksat_hz.total <- (sum( ksat_hz , na.rm=TRUE))/ (sum((thick *((!is.na( ksat_hz))*1)),na.rm=TRUE))                    # depth weighted average of ksat each component  key
  wcthirdbar.total <- (sum( wcthirdbar , na.rm=TRUE))/ (sum((thick *((!is.na( wcthirdbar))*1)),na.rm=TRUE))           # depth weighted average of water content at 1/3 bar for each component  key
  wctfifteendbar.total <- (sum(wctfifteendbar, na.rm=TRUE))/(sum((thick *((!is.na( wctfifteendbar))*1)),na.rm=TRUE))  # depth weighted average of water content at 15 bar for each component  key
  dbthirdbar.total <- (sum(dbthirdbar, na.rm=TRUE))/ (sum((thick *((!is.na( dbthirdbar))*1)),na.rm=TRUE))             # depth weighted average of bulk density  at 1/3 bar for each component  key
  sand.total <- (sum(sand, na.rm=TRUE))/ (sum((thick *((!is.na( sand))*1)),na.rm=TRUE))                               # depth weighted average of sand   for each component  key
  clay.total <- (sum(clay, na.rm=TRUE))/ (sum((thick *((!is.na(clay))*1)),na.rm=TRUE))                                # depth weighted average of clay  for each component  key
  om.total <- (sum(om, na.rm=TRUE))/ (sum((thick *((!is.na( om))*1)),na.rm=TRUE))                                     # depth weighted average of organic matter  for each component  key


  data.frame(wcthird=wcthirdbar.total,wctfifteendbar=wctfifteendbar.total,  dbthirdbar=dbthirdbar.total,
             sand=sand.total,clay=clay.total,om=om.total,awc=awc.total,awcdepth=awc.depth,thick=abs(thick.total) , wt=wt,ksat_hz=ksat_hz.total) # return profile water storage and component pct

  #,f=fval
}

# function for copmuting weighted-mean whc within a map unit
mu.mean.whc <- function(i) {
  thick <- wtd.mean(i$ thick, weights=i$wt) # safely compute wt. mean ksat for each map unit key
  # ksat <- wtd.mean(i$ ksat, weights=i$wt) # safely compute wt. mean ksat for each map unit key

  ksat_hz<- wtd.mean(i$ksat_hz, weights=i$wt) # safely compute wt. mean water content at 1/3 bar for each map unit key
  wcthird<- wtd.mean(i$wcthird, weights=i$wt) # safely compute wt. mean water content at 1/3 bar for each map unit key
  wctfifteendbar <- wtd.mean(i$wctfifteendbar, weights=i$wt) # safely compute wt. mean water content at 15 bar for each map unit key
  dbthirdbar <- wtd.mean(i$dbthirdba, weights=i$wt) # safely compute wt. mean bulk density at 1/3 bar for each map unit key
  sand <- wtd.mean(i$sand, weights=i$wt) # safely compute wt. mean sand for each map unit key
  clay<- wtd.mean(i$ clay, weights=i$wt) # safely compute wt. mean clay for each map unit key
  om<- wtd.mean(i$om, weights=i$wt) # safely compute wt. mean organic matter for each map unit key
  # fvalue= wtd.mean(i$f, weights=i$wt,na.rm=TRUE)
  # ts= wtd.mean(i$tr, weights=i$wt,na.rm=TRUE)

  data.frame(depth=thick,wcthird=wcthird,wctfifteendbar=wctfifteendbar, dbthirdbar= dbthirdbar,
             sand=sand,clay=clay,om=om, ksat_hz=ksat_hz) # return wt. mean water storage
  #,fval=fvalue*-1
}

print (head(res))

print ('Changin NA values to avg')
# give an average value for NA items. For some reason, looks like it does not work if no NA value
for(i in 1:ncol(res)){

  res[is.na(res[,i]), i] <- lapply(res[,i], mean, na.rm = TRUE)  #colMeans(res[,i], na.rm = TRUE) # mean(res[,i], na.rm = TRUE)
}
print (head(res))


# aggregate by component  unit
co.whc <- ddply(res, c('mukey', 'cokey'), co.mean.whc)
print ("5. Weighted average by component completed successfully" )

# aggregate by map unit
mu.whc <- ddply(co.whc, 'mukey', mu.mean.whc)


print ("5. Weighted average by map units successful" )
write.csv(mu.whc, 'mu.csv')
write.csv(co.whc, 'co.csv')


# drainable moisture content
porosity=1-mu.whc$dbthirdbar/2.65
dth1=porosity-mu.whc$wcthird/100

# plant available moisture content
dth2=(mu.whc$wcthird-mu.whc$wctfifteendbar)/100
dth2[dth2<0]=0

#soil depth
#soildepth=res2$brockdepmin

# pore disconnectedness index
b=(log(1500)-log(33))/(log(mu.whc$wcthird)-log(mu.whc$wctfifteendbar)) ## from Rawsl 1992 et al
c1=1; ## TOPNET use c=1
c=2*b+3 # from Dingman
b=(c-3)/2

##Green-Ampt wetting front suction parameter in m
##equations are taking from Rawls et at 1992
suctiont1=-21.67*mu.whc$sand/100-27.93*mu.whc$clay/100-81.97*dth1+71.12*(mu.whc$sand*dth1/100) +8.29*(mu.whc$clay*dth1/100)+14.05*(mu.whc$sand*mu.whc$clay/10000)+27.16
suction=suctiont1+(0.02*suctiont1*suctiont1-0.113*suctiont1-0.70) # unit kpa
suction_meter=suction*0.102                                       # convert kpa=m of water
psif=((2*b+3)/(2*b+6))*abs(suction_meter)                         # equation from Dingman. Gives negative values too. Needs correction



BBL = suction_meter*1000                                   # bubbling pressure, in mm
BBL[BBL<72]=72                                             # FOR THE TIME BEING, TO MAKE THE PROGRAM WORK! THIS IS NOT ACCEPTABLE TO IMPLEMENT
PSD = 1/b                                                  # pore_size_distribution
RSM = mu.whc$wctfifteendbar/100                            # residual soil moisture content
SSM = porosity
ksat_rawls = 1930* (porosity - mu.whc$wcthird/100)^(3-PSD) # mm/hr
ksat_rawls = ksat_rawls /3600                              # mm/s
ksat_hz = mu.whc$ksat_hz / 1000                            # mm/s. convert from micrometer/s (as required by PyTOPKPAI)

soil_data=data.frame(mukey=mu.whc$mukey, depth=mu.whc$depth, dth1=dth1,dth2=dth2,
                     psif=psif, BBL=BBL, PSD=PSD, RSM=RSM, SSM=SSM, ksat_rawls=ksat_rawls, ksat_hz=ksat_hz)



names(soil_data)[1]='ID'
names(mu)[1]='ID'
soildata_join=join( mu,soil_data, by='ID')
rat.new=join(rat,soildata_join,type='left')
rat.new <- rat.new[,c("ID", "COUNT","dth1","dth2","psif", "depth",         "BBL", "PSD", "RSM", "SSM", "ksat_rawls", "ksat_hz")]




levels(r)=rat.new

q=c(args[3],args[4],args[5],args[6],args[7],args[8],args[9],     args[10],args[11],args[12],args[13] )

print ('6. Creating soil Rasters')
#setwd(args[3])
for(i in 1:length(q)){
  r.new=deratify(r,att=names( rat.new)[i+2])
  r.new[is.na(r.new[])] <- cellStats(r.new,mean) ## fill missing data with mean value
  writeRaster(r.new,q[i],overwrite=TRUE,datatype='FLT4S',format="GTiff",options="COMPRESS=NONE")
  print (i)
}




