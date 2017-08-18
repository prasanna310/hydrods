# Authors: Prasanna Dahal

args<-commandArgs(TRUE)

#args = c('C:/Optiplex_960/TOPNET/test_folder/test2','Soil_mukey.tif',
#        'dth1.tif', 'dth2.tif', 'psif.tif', 'sd.tif',
#        'bbl.tif', 'psd.tif', 'rsm.tif', 'ssm.tif', 'ksat_LUT.tif', 'ksat_ssurgo_wtd.tif', 'ksat_ssurgo_min.tif', 'hydrogrp.tif',
#        '-117.2250812006314', '-117.01960618176219', '34.08241754878665', '34.23415294733622'  , '162', '137' ,
#        'C:/Users/Prasanna/Desktop/alternateDownlaods/reference_raster.tif') # xmin,xmax,ymin,ymax, ,ncol, nrow

suppressMessages(library(raster))
suppressMessages(library(plyr))
suppressMessages(library(Hmisc))
suppressMessages(library(soilDB))
suppressMessages(library(SSOAP))
suppressMessages(library(foreign))
suppressMessages(library(shapefiles))
suppressMessages(library(rgdal))
suppressMessages(library(sp))
suppressMessages(library(rgeos))
#suppressMessages(library(sqldf))

print (args)
print ("1. All the packages loaded" )

dir=args[1]
setwd(dir)

df_lut = read.csv('/home/ahmet/ciwater/usu_data_service/pytopkapi_data_service/GREENAMPT_LOOKUPTABLE.csv')
#df_lut = read.csv('C:/Optiplex_960/TOPNET/test_folder/working_folder/GREENAMPT_LOOKUPTABLE.csv')

xmin=as.numeric(args[15])
xmax = as.numeric(args[16])
ymin=as.numeric(args[17])
ymax =as.numeric(args[18])

b <- c(as.numeric(args[15]), as.numeric(args[16]),as.numeric(args[17]), as.numeric(args[18]) )  # b <- c(-120.9, -120.8, 37.7, 37.8), # bounding box: xmin, xmax, ymin, ymax
# convert bounding box to WKT
p <- writeWKT(as(extent(b), 'SpatialPolygons'))



query_mukey <- paste0("SELECT mukey, muname
            FROM mapunit
            WHERE mukey IN (
            SELECT * from SDA_Get_Mukey_from_intersection_with_WktWgs84('", p, "')
            )")

res_mukey <- SDA_query(query_mukey)


in.statement <- format_SQL_in_statement(unique(res_mukey$mukey)) #unique(x$MUKEY)


##make query for each variable
print ("2. about to make a query" )

q1 <- paste("SELECT component.mukey, component.cokey, compname, comppct_r,
            hzdept_r, hzdepb_r, hzname,awc_r, ksat_r,wthirdbar_r,wfifteenbar_r,dbthirdbar_r,sandtotal_r,claytotal_r,om_r,
            texcl
            FROM component
            JOIN chorizon ON component.cokey = chorizon.cokey
            join chtexturegrp on chtexturegrp.chkey = chorizon.chkey
            join chtexture on chtexture.chtgkey=chtexturegrp.chtgkey
            AND mukey IN ", in.statement, "ORDER BY mukey, comppct_r DESC, hzdept_r ASC", sep="")

# get component and horizon-level data for these map unit keys
res1 <- SDA_query(q1)

print ("3. Querry successful" )
write.csv(res1,'Response_SDA_query.csv')


# # query for soil depth. Almost always results NA values for most MUKEY. BUT, hydrologic soil group is useful from this data.
q2 <- paste("SELECT mukey,brockdepmin, hydgrpdcd
            FROM muaggatt
            WHERE  mukey IN ", in.statement, "
            ", sep="")
res2 <- SDA_query(q2)
write.csv(res2,'02_Response_SDA_query2.csv')




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
  depth <- max(abs(i$hzdepb_r/100))   # depth in meters
  #ksat <- thick/i$ksat_r #compute saturated hydraulic conductivity
  #ksat<-i$ksat_r[1]*3600*10^(-6)

  ksat_ssurgo_wtd <- thick * i$ksat_r                          # compute ksat
  ksat_ssurgo_min <- min(i$ksat_r)                          # compute ksat, min of the horizon

  wcthirdbar <- thick * i$wthirdbar_r                  # compute water content at 1/3 bar with horizon
  wctfifteendbar  <- thick * i$wfifteenbar_r           # compute water content at 15 bar with horizon
  dbthirdbar  <- thick * i$dbthirdbar_r                # compute density at 1/3 bar with horizon
  sand <- thick * i$sandtotal_r                        # compute percentage of sand by weight with horizon
  clay<- thick * i$claytotal_r                         # compute percentage of clay  by weight with horizon
  om <- thick * i$om_r                                 # compute percentage of organic matter by weight  with horizon
  awc <-  i$awc_r                                      # available water capacity

  # values from Lookuptables
  ksat_LUT <- thick * i$ksat_LUT                             # ksat, but from LUT joint to horizon level dataset
  Porosity_LUT <- thick * i$Porosity_LUT                     # porosity, but from LUT joint to horizon level dataset
  RSM_LUT <- thick * i$RSM_LUT                               # residual soil moisture, but from LUT joint to horizon level dataset
  BBL_LUT <- thick * i$BBL_LUT                               # bubbling pressure, but from LUT joint to horizon level dataset
  PSD_LUT <- thick * i$PSD_LUT                               # poresize distribution, but from LUT joint to horizon level dataset


  thick.total=sum(thick, na.rm=TRUE)
  awc.total=sum(awc, na.rm=TRUE)
  awc.depth=( sum((thick *((!is.na(awc))*1)),na.rm=TRUE))
  #ksat.total <- sum(thick, na.rm=TRUE)/(sum(ksat, na.rm=TRUE)) # Harmonic mean
  #ksat.total<-ksat

  ksat_ssurgo_wtd.total <- (sum( ksat_ssurgo_wtd , na.rm=TRUE))/ (sum((thick *((!is.na( ksat_ssurgo_wtd))*1)),na.rm=TRUE))                    # depth weighted average of ksat each component  key
  wcthirdbar.total <- (sum( wcthirdbar , na.rm=TRUE))/ (sum((thick *((!is.na( wcthirdbar))*1)),na.rm=TRUE))           # depth weighted average of water content at 1/3 bar for each component  key
  wctfifteendbar.total <- (sum(wctfifteendbar, na.rm=TRUE))/(sum((thick *((!is.na( wctfifteendbar))*1)),na.rm=TRUE))  # depth weighted average of water content at 15 bar for each component  key
  dbthirdbar.total <- (sum(dbthirdbar, na.rm=TRUE))/ (sum((thick *((!is.na( dbthirdbar))*1)),na.rm=TRUE))             # depth weighted average of bulk density  at 1/3 bar for each component  key
  sand.total <- (sum(sand, na.rm=TRUE))/ (sum((thick *((!is.na( sand))*1)),na.rm=TRUE))                               # depth weighted average of sand   for each component  key
  clay.total <- (sum(clay, na.rm=TRUE))/ (sum((thick *((!is.na(clay))*1)),na.rm=TRUE))                                # depth weighted average of clay  for each component  key
  om.total <- (sum(om, na.rm=TRUE))/ (sum((thick *((!is.na( om))*1)),na.rm=TRUE))                                     # depth weighted average of organic matter  for each component  key

  # values from Lookuptables
  ksat_LUT.total <- (sum(ksat_LUT, na.rm=TRUE))/ (sum((thick *((!is.na(ksat_LUT))*1)),na.rm=TRUE))                       # depth weighted average of organic matter  for each component  key
  Porosity_LUT.total <- (sum(Porosity_LUT, na.rm=TRUE))/ (sum((thick *((!is.na( Porosity_LUT))*1)),na.rm=TRUE))          # depth weighted average of organic matter  for each component  key
  RSM_LUT.total <- (sum(RSM_LUT, na.rm=TRUE))/ (sum((thick *((!is.na( RSM_LUT))*1)),na.rm=TRUE))                         # depth weighted average of organic matter  for each component  key
  BBL_LUT.total <- (sum(BBL_LUT, na.rm=TRUE))/ (sum((thick *((!is.na( BBL_LUT))*1)),na.rm=TRUE))                         # depth weighted average of organic matter  for each component  key
  PSD_LUT.total <- (sum(PSD_LUT))/ (sum((thick *((!is.na( PSD_LUT))*1)),na.rm=TRUE))                                     # depth weighted average of organic matter  for each component  key


  data.frame(wcthird=wcthirdbar.total,wctfifteendbar=wctfifteendbar.total,  dbthirdbar=dbthirdbar.total,
             sand=sand.total,clay=clay.total,om=om.total,awc=awc.total,awcdepth=awc.depth,thick=abs(thick.total) , wt=wt,
             ksat_ssurgo_wtd=ksat_ssurgo_wtd.total, ksat_ssurgo_min = ksat_ssurgo_min,
             ksat_LUT=ksat_LUT.total,Porosity_LUT= Porosity_LUT.total, RSM_LUT=RSM_LUT.total,BBL_LUT=BBL_LUT.total,PSD_LUT=PSD_LUT.total,
             depth= depth
  ) # return profile water storage and component pct

  #,f=fval
}

# function for copmuting weighted-mean whc within a map unit
mu.mean.whc <- function(i) {
  thick <- wtd.mean(i$ thick, weights=i$wt) # safely compute wt. mean ksat for each map unit key
  # ksat <- wtd.mean(i$ ksat, weights=i$wt) # safely compute wt. mean ksat for each map unit key

  depth <- max(i$depth)

  ksat_ssurgo_wtd<- wtd.mean(i$ksat_ssurgo_wtd, weights=i$wt) # safely compute wt. mean for each map unit key
  ksat_ssurgo_min <- wtd.mean(i$ksat_ssurgo_min, weights=i$wt) # safely compute wt. mean for each map unit key

  wcthird<- wtd.mean(i$wcthird, weights=i$wt) # safely compute wt. mean water content at 1/3 bar for each map unit key
  wctfifteendbar <- wtd.mean(i$wctfifteendbar, weights=i$wt) # safely compute wt. mean water content at 15 bar for each map unit key
  dbthirdbar <- wtd.mean(i$dbthirdba, weights=i$wt) # safely compute wt. mean bulk density at 1/3 bar for each map unit key
  sand <- wtd.mean(i$sand, weights=i$wt) # safely compute wt. mean sand for each map unit key
  clay<- wtd.mean(i$ clay, weights=i$wt) # safely compute wt. mean clay for each map unit key
  om<- wtd.mean(i$om, weights=i$wt) # safely compute wt. mean organic matter for each map unit key
  # fvalue= wtd.mean(i$f, weights=i$wt,na.rm=TRUE)
  # ts= wtd.mean(i$tr, weights=i$wt,na.rm=TRUE)

  # values from Lookuptables
  ksat_LUT<- wtd.mean(i$ksat_LUT, weights=i$wt) # safely compute wt. mean organic matter for each map unit key
  Porosity_LUT<- wtd.mean(i$Porosity_LUT, weights=i$wt) # safely compute wt. mean organic matter for each map unit key
  RSM_LUT<- wtd.mean(i$RSM_LUT, weights=i$wt) # safely compute wt. mean organic matter for each map unit key
  BBL_LUT<- wtd.mean(i$BBL_LUT, weights=i$wt) # safely compute wt. mean organic matter for each map unit key
  PSD_LUT<- wtd.mean(i$PSD_LUT, weights=i$wt) # safely compute wt. mean organic matter for each map unit key


  data.frame(depth=depth,wcthird=wcthird,wctfifteendbar=wctfifteendbar, dbthirdbar= dbthirdbar,
             sand=sand,clay=clay,om=om, ksat_ssurgo_wtd=ksat_ssurgo_wtd, ksat_ssurgo_min = ksat_ssurgo_min,
             Porosity_LUT=Porosity_LUT,ksat_LUT=ksat_LUT,Porosity_LUT=Porosity_LUT,RSM_LUT=RSM_LUT, BBL_LUT=BBL_LUT ,PSD_LUT=PSD_LUT
  ) # return wt. mean water storage
  #,fval=fvalue*-1
}

res = merge(x = res1, y = df_lut, by = "texcl")
write.csv(res,'03_texture_joint_df.csv')


print ('Changin NA values to avg')
# give an average value for NA items. For some reason, looks like it does not work if no NA value
for(i in 1:ncol(res)){
  res[is.na(res[,i]), i] <- lapply(res[,i], mean, na.rm = TRUE)  #colMeans(res[,i], na.rm = TRUE) # mean(res[,i], na.rm = TRUE)
}



# aggregate by component  unit
co.whc <- ddply(res, c('mukey', 'cokey'), co.mean.whc)
print ("5. Weighted average by component completed successfully" )

# aggregate by map unit
mu1.whc <- ddply(co.whc, 'mukey', mu.mean.whc)

mu.whc = merge(x = mu1.whc, y = res2, by = "mukey")

print ("5. Weighted average by map units successful" )
write.csv(co.whc, '04_component_agg_df.csv')
write.csv(mu.whc, '05_mapunit_agg_df.csv')


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
ksat_ssurgo_wtd = mu.whc$ksat_ssurgo_wtd / 1000                    # mm/s. convert from micrometer/s (as required by PyTOPKPAI)
ksat_ssurgo_min = mu.whc$ksat_ssurgo_min / 1000

# reclassify soil hydrological group names
mu.whc$hydgrpdcd[mu.whc$hydgrpdcd=='A']<- 10.0
mu.whc$hydgrpdcd[mu.whc$hydgrpdcd=='B']<- 20.0
mu.whc$hydgrpdcd[mu.whc$hydgrpdcd=='C']<- 30.0
mu.whc$hydgrpdcd[mu.whc$hydgrpdcd=='D']<- 40.0

mu.whc$hydgrpdcd[mu.whc$hydgrpdcd=='A/D']<- 14.0
mu.whc$hydgrpdcd[mu.whc$hydgrpdcd=='B/D']<- 24.0
mu.whc$hydgrpdcd[mu.whc$hydgrpdcd=='C/D']<- 34.0

mu.whc$hydgrpdcd[mu.whc$hydgrpdcd=='A/B']<- 12.0
mu.whc$hydgrpdcd[mu.whc$hydgrpdcd=='C/B']<- 32.0
mu.whc$hydgrpdcd[mu.whc$hydgrpdcd=='D/B']<- 42.0

mu.whc$hydgrpdcd[mu.whc$hydgrpdcd=='A/C']<- 13.0
mu.whc$hydgrpdcd[mu.whc$hydgrpdcd=='B/C']<- 23.0
mu.whc$hydgrpdcd[mu.whc$hydgrpdcd=='D/C']<- 33.0

mu.whc$hydgrpdcd[mu.whc$hydgrpdcd=='B/A']<- 21.0
mu.whc$hydgrpdcd[mu.whc$hydgrpdcd=='C/A']<- 31.0
mu.whc$hydgrpdcd[mu.whc$hydgrpdcd=='D/A']<- 41.0
mu.whc$hydgrpdcd <- as.numeric(mu.whc$hydgrpdcd)

# SOME INFO IN DUAL HYDROLOGIC SOIL GROUPS
# dual hydrologic soil groups (A/D, B/D, and C/D)based on their saturated hydraulic conductivity and
# the water table depth when drained. The first letter applies to the drained condition and the second to the
# undrained condition. For the purpose of hydrologic soil group, adequately drained means that the seasonal
# high water table is kept at least 60 centimeters [24inches] below the surface in a soil where it would be higher in a natural state




# old soil_data df,
# prepared from calculating values using Rawls equations
# soil_data=data.frame(mukey=mu.whc$mukey, depth=mu.whc$depth, dth1=dth1,dth2=dth2,
#                      psif=psif, BBL=BBL, PSD=PSD, RSM=RSM, SSM=SSM, ksat_rawls=ksat_rawls, ksat_hz=ksat_hz)


# new soil_data df,
# prepared from using LUT to calculate some parameters like: PSD, RSM, BBL, KSAT
# this datafram should not contain unnecessary fields, as the fields index is used to create raster file below
soil_data=data.frame(mukey=mu.whc$mukey, depth=mu.whc$depth, dth1=dth1,dth2=dth2,psif=psif,
                     BBL=mu.whc$BBL_LUT, PSD=mu.whc$PSD_LUT, RSM=mu.whc$RSM_LUT, SSM=SSM, ksat_LUT=mu.whc$ksat_LUT, ksat_ssurgo_wtd=ksat_ssurgo_wtd,
                     ksat_ssurgo_min=ksat_ssurgo_min,hydrogrp=mu.whc$hydgrpdcd )





# *********************************** #
# for older R version, downloading is not supported. That is why the function mapunit_geom_by_ll_bbox  was changed :
mapunit_geom_by_ll_bbox2 <- function (bbox, source = "sda")
{
  if (!requireNamespace("rgdal", quietly = TRUE))
    stop("please install the `rgdal` package", call. = FALSE)
  bbox.text <- paste(bbox, collapse = ",")
  ogr.Drv <- rgdal::ogrDrivers()$name
  if (source == "sda") {
    if (!"GML" %in% ogr.Drv)
      stop("GML support is missing from your GDAL/OGR build.",
           call. = FALSE)
    u <- paste("https://sdmdataaccess.nrcs.usda.gov/Spatial/SDMNAD83Geographic.wfs?Service=WFS&Version=1.0.0&Request=GetFeature&Typename=MapunitPoly&BBOX=",
               bbox.text, sep = "")
    file.extension <- ".gml"
    file.layer <- "mapunitpoly"
  }
  if (source == "soilweb") {
    stop("Data from SoilWeb is currently not supported.",
         call. = FALSE)
    if (!"KML" %in% ogr.Drv)
      stop("KML support is missing from your GDAL/OGR build.",
           call. = FALSE)
    u <- paste("https://casoilresource.lawr.ucdavis.edu/soil_web/export.php?format=kml&srid=4326&BBOX=",
               bbox.text, sep = "")
    file.extension <- ".kml"
    file.layer <- "Soil Polygons"
  }
  # td <- tempdir()
  # tf <- tempfile(pattern = "file", tmpdir = td)
  tf.full <- paste('temp_vec', file.extension, sep = "")
  print (c('temp_file= ',tf.full ))
  download.file(url = u, destfile = tf.full, quiet = FALSE,cacheOK=FALSE, method='auto')  #method="curl" ) #https://stackoverflow.com/questions/19890633/r-produces-unsupported-url-scheme-error-when-getting-data-from-https-sites
  print (c('Is downloaded file present? ', file.exists(path.expand(tf.full))))
  d <- rgdal::readOGR(dsn = path.expand(tf.full), layer = file.layer, disambiguateFIDs = TRUE,stringsAsFactors = FALSE)
  d$gml_id <- NULL
  gc()
  unlink(tf.full)
  return(d)
}

# *********************************** #


# now, get the geometry data in the form of vector data (shapefile kindof data)
bbox <- c(xmin,ymin,xmax,ymax)                                           # a <- c(-120.54,38.61,-120.41,38.70)
print  (c('xmin, ymin, xmax, ymax',': ',  bbox))
res_geometry <- mapunit_geom_by_ll_bbox2(bbox)
#res_geometry_prj = spTransform(res_geometry, CRS("+init=epsg:4326"))    # epsg:4326=WGS geographic, epsg:102003=NA Albers Equal Area conic
# res_geometry_prj = spTransform(res_geometry,CRS("+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=37.5 +lon_0=-96 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"))      # This is probably how you define epsg:102003
reference_raster = raster(args[21])
res_geometry_prj = spTransform(res_geometry,proj4string(reference_raster) )




new_raster = raster(args[21])
ras2 <- rasterize(res_geometry_prj, new_raster, field='mukey' , fun='first' )
writeRaster(ras2, "Soil_mukey.tif", format = "GTiff", overwrite=TRUE)



# # make a blank raster, and have the values of shapefile there
# # source: https://gis.stackexchange.com/questions/158159/snapping-raster-grids-in-r
# r <- raster(ncol=as.numeric(args[19]), nrow=as.numeric(args[20]) )
# r.new = resample(r, reference_raster, "bilinear")                         # Resample to same grid:
# r2 = crop(r.new, reference_raster)
# r.new = mask(r.new, r2)                                                   # Removed data which falls outside one of the rasters
# extent(r) <- extent(res_geometry_prj)                                     # make new rasters extent same as data, i.e. from mapunit_geom_by_ll_bbox
# ras <- rasterize(res_geometry_prj, r, field='mukey' , fun='first' )
# writeRaster(ras, "Soil_mukey2.tif", format = "GTiff", overwrite=TRUE)
#





working_dir = "."

for(i in 1:length(soil_data)){
  temp_df = soil_data[,c(1,i)]
  write.table( temp_df, paste(working_dir, '/', names(temp_df)[2],'.csv', sep=""), sep=",", row.names=FALSE, col.names=FALSE)
  system(paste("python3.4 /home/ahmet/ciwater/usu_data_service/pytopkapi_data_service/reclassifyraster.py",'-i', 'Soil_mukey.tif', '-lut',paste(working_dir, '/', names(temp_df)[2],'.csv', sep="") , '-o', paste(names(temp_df)[2],'.tif', sep=""), sep=" ")  )
}


