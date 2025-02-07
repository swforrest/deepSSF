### title: "Preparing data for DL model fitting"
### author: "Scott Forrest"
  
## Loading packages
library(tidyverse)
packages <- c("amt", "sf", "terra", "beepr", "tictoc")
walk(packages, require, character.only = T)

## Import data and clean

buffalo <- read_csv("data/buffalo.csv")

# remove individuals that have poor data quality or less than about 3 months of data. 
# The "2014.GPS_COMPACT copy.csv" string is a duplicate of ID 2024, so we exlcude it
buffalo <- buffalo %>% filter(!node %in% c("2014.GPS_COMPACT copy.csv", 
                                           2029, 2043, 2265, 2284, 2346))

buffalo <- buffalo %>%  
  group_by(node) %>% 
  arrange(DateTime, .by_group = T) %>% 
  distinct(DateTime, .keep_all = T) %>% 
  arrange(node) %>% 
  mutate(ID = node)

buffalo_clean <- buffalo[, c(12, 2, 4, 3)]
colnames(buffalo_clean) <- c("id", "time", "lon", "lat")
attr(buffalo_clean$time, "tzone") <- "Australia/Queensland"
head(buffalo_clean)
tz(buffalo_clean$time)

buffalo_ids <- unique(buffalo_clean$id)

## Setup trajectory
# Use the `amt` package to create a trajectory object from the cleaned data. 

buffalo_all <- buffalo_clean %>% mk_track(id = id,
                                          lon,
                                          lat, 
                                          time, 
                                          all_cols = T,
                                          crs = 4326) %>% 
  transform_coords(crs_to = 3112, crs_from = 4326) # Transformation to GDA94 / 
# Geoscience Australia Lambert (https://epsg.io/3112)

buffalo_all <- buffalo_all %>% arrange(id)
# nest the data by individual
buffalo_all_nested <- buffalo_all %>% arrange(id) %>% nest(data = -"id")


### Reading in the environmental covariates

ndvi_projected <- rast("mapping/cropped rasters/ndvi_GEE_projected_watermask20230207.tif")
terra::time(ndvi_projected) <- as.POSIXct(lubridate::ymd("2018-01-01") + months(0:23))
slope <- rast("mapping/cropped rasters/slope_raster.tif")
veg_herby <- rast("mapping/cropped rasters/veg_herby.tif")
canopy_cover <- rast("mapping/cropped rasters/canopy_cover.tif")

# change the names (these will become the column names when extracting 
# covariate values at the used and random steps)
names(ndvi_projected) <- rep("ndvi", terra::nlyr(ndvi_projected))
names(slope) <- "slope"
names(veg_herby) <- "veg_herby"
names(canopy_cover) <- "canopy_cover"


# Generating the data to fit the ML models

res <- terra::res(ndvi_projected)[1]

# how much to trim on either side of the location
buffer <- 1250 + (res/2)
nxn_cells <- buffer*2/res # should be 101 cells in this case, so that there is a centre cell

# hourly lag
hourly_lag <- 1

# subset the data
n_samples <- 10
# n_samples <- nrow(buffalo_all)

# if subsetting the data by individual or number of samples
# buffalo_data <- buffalo_all %>% filter(id == "2005") %>% slice(1:n_samples)
buffalo_data <- buffalo_all %>% slice(1:n_samples)

tic()

buffalo_data_covs <- buffalo_data %>% mutate(
  
  x1_ = x_,
  y1_ = y_,
  x2_ = lead(x1_, n = hourly_lag, default = NA),
  y2_ = lead(y1_, n = hourly_lag, default = NA),
  x2_cent = x2_ - x1_,
  y2_cent = y2_ - y1_,
  t2_ = lead(t_, n = hourly_lag, default = NA),
  t_diff = round(difftime(t2_, t_, units = "hours"),0),
  hour_t1 = lubridate::hour(t_),
  yday_t1 = lubridate::yday(t_),
  hour_t2 = lubridate::hour(t2_),
  hour_t2_sin = sin(2*pi*hour_t2/24),
  hour_t2_cos = cos(2*pi*hour_t2/24),
  yday_t2 = lubridate::yday(t2_),
  yday_t2_sin = sin(2*pi*yday_t2/365),
  yday_t2_cos = cos(2*pi*yday_t2/365),
  
  sl = c(sqrt(diff(y_)^2 + diff(x_)^2), NA),
  log_sl = log(sl),
  bearing = c(atan2(diff(y_), diff(x_)), NA),
  # bearing_degrees = bearing * 180/pi %% 360,
  bearing_sin = sin(bearing),
  bearing_cos = cos(bearing),
  # ta = c(NA, ifelse(diff(bearing) > pi, diff(bearing) - 2 * pi, diff(bearing))),
  ta = c(NA, ifelse(
    diff(bearing) > pi, diff(bearing)-(2*pi), ifelse(
      diff(bearing) < -pi, diff(bearing)+(2*pi), diff(bearing)))),
  cos_ta = cos(ta),
  
  # extent for cropping the spatial covariates
  x_min = x_ - buffer,
  x_max = x_ + buffer,
  y_min = y_ - buffer,
  y_max = y_ + buffer,
  
) %>% rowwise() %>% mutate(
  
  extent_00centre = list(ext(x_min - x_, x_max - x_, y_min - y_, y_max - y_)),
  
  # NDVI
  ndvi_index = which.min(abs(difftime(t_, terra::time(ndvi_projected)))),
  ndvi_local = list(crop(ndvi_projected[[ndvi_index]],
                         ext(x_min, x_max, y_min, y_max))),
  ndvi_cent = list({
    ndvi_cent <- rep(ndvi_local)
    ext(ndvi_cent) <- extent_00centre
    ndvi_cent
  }),
  
  # herbaceous vegetation
  veg_herby_local = list(crop(veg_herby, ext(x_min, x_max, y_min, y_max))),
  veg_herby_cent = list({
    veg_herby_cent = crop(veg_herby, ext(x_min, x_max, y_min, y_max))
    # veg_herby_cent <- rep(veg_herby_local)
    ext(veg_herby_cent) <- extent_00centre
    veg_herby_cent
  }),
  
  # canopy cover
  canopy_cover_local = list(crop(canopy_cover, ext(x_min, x_max, y_min, y_max))),
  canopy_cover_cent = list({
    canopy_cover_cent = crop(canopy_cover, ext(x_min, x_max, y_min, y_max))
    # canopy_cover_cent <- rep(canopy_cover_local)
    ext(canopy_cover_cent) <- extent_00centre
    canopy_cover_cent
  }),
  
  # slope
  slope_local = list(crop(slope, ext(x_min, x_max, y_min, y_max))),
  slope_cent = list({
    slope_cent <- crop(slope, ext(x_min, x_max, y_min, y_max))
    # slope_cent <- rep(slope_local)
    ext(slope_cent) <- extent_00centre
    slope_cent
  }),
  
  # rasterised location of the next step
  points_vect_local = list(terra::vect(cbind(x2_, y2_), type = "points", crs = "EPSG:3112")),
  pres_local = list(rasterize(points_vect_local, ndvi_local, background=0)),
  
  # rasterised location of the next step - centred on (0,0)
  points_vect_cent = list(terra::vect(cbind(x2_ - x_, y2_ - y_), type = "points", crs = "EPSG:3112")),
  pres_cent = list(rasterize(points_vect_cent, ndvi_cent, background=0))
  
) %>% ungroup() # to remove the 'rowwise' class

toc()

buffalo_data_covs
which(is.na(buffalo_data_covs$ta))


## Plot the covariates

n_plots <- 5 # for each covariate

walk(buffalo_data_covs$ndvi_cent[1:n_plots], plot)
walk(buffalo_data_covs$veg_herby_cent[1:n_plots], terra::plot)
walk(buffalo_data_covs$canopy_cover_cent[1:n_plots], terra::plot)
walk(buffalo_data_covs$slope_cent[1:n_plots], terra::plot)
walk(buffalo_data_covs$pres_cent[1:n_plots], terra::plot)

saveRDS(buffalo_data_covs, paste0("buffalo_data_df_", nxn_cells, "x", nxn_cells, 
                                  "_lag_", hourly_lag, "hr_n", n_samples, ".rds"))


# Filter when the next step is outside the extent and drop NAs from the turning angle column

buffalo_data_covs <- buffalo_data_covs %>% filter(x2_cent > -buffer & x2_cent < buffer & y2_cent > -buffer & y2_cent < buffer) %>% drop_na(ta)
buffalo_data_covs


# Export objects

nxn_cells # should be 101

buffalo_data_df <- buffalo_data_covs %>%
  dplyr::select(-extent_00centre, 
                -ndvi_local, -ndvi_cent, 
                -veg_herby_local, -veg_herby_cent, 
                -canopy_cover_local, -canopy_cover_cent,
                -slope_local, -slope_cent,
                -points_vect_local, -points_vect_cent, 
                -pres_local, -pres_cent)

# to save the dataframe without the raster objects
write_csv(buffalo_data_df, paste0("buffalo_data_df_lag_", hourly_lag, "hr_n", n_samples, ".csv"))

# rast(buffalo_data_covs$ndvi_local) %>% 
#   writeRaster(paste0("buffalo_ndvi_local", nxn_cells, "x", nxn_cells, "_lag_", hourly_lag, "hr_n", n_samples, ".tif"), overwrite = T)
# rast(buffalo_data_covs$veg_herby_local) %>% 
#   writeRaster(paste0("buffalo_herby_local", nxn_cells, "x", nxn_cells, "_lag_", hourly_lag, "hr_n", n_samples, ".tif"), overwrite = T)
# rast(buffalo_data_covs$canopy_cover_local) %>% 
#   writeRaster(paste0("buffalo_canopy_local", nxn_cells, "x", nxn_cells, "_lag_", hourly_lag, "hr_n", n_samples, ".tif"), overwrite = T)
# rast(buffalo_data_covs$slope_local) %>% 
#   writeRaster(paste0("buffalo_slope_local", nxn_cells, "x", nxn_cells, "_lag_", hourly_lag, "hr_n", n_samples, ".tif"), overwrite = T)
# rast(buffalo_data_covs$pres_local) %>% 
#   writeRaster(paste0("buffalo_pres_local", nxn_cells, "x", nxn_cells, "_lag_", hourly_lag, "hr_n", n_samples, ".tif"), overwrite = T)

rast(buffalo_data_covs$ndvi_cent) %>% 
  writeRaster(paste0("buffalo_ndvi_cent", nxn_cells, "x", nxn_cells, "_lag_", hourly_lag, "hr_n", n_samples, ".tif"), overwrite = T)
rast(buffalo_data_covs$veg_herby_cent) %>% 
  writeRaster(paste0("buffalo_herby_cent", nxn_cells, "x", nxn_cells, "_lag_", hourly_lag, "hr_n", n_samples, ".tif"), overwrite = T)
rast(buffalo_data_covs$canopy_cover_cent) %>% 
  writeRaster(paste0("buffalo_canopy_cent", nxn_cells, "x", nxn_cells, "_lag_", hourly_lag, "hr_n", n_samples, ".tif"), overwrite = T)
rast(buffalo_data_covs$slope_cent) %>% 
  writeRaster(paste0("buffalo_slope_cent", nxn_cells, "x", nxn_cells, "_lag_", hourly_lag, "hr_n", n_samples, ".tif"), overwrite = T)
rast(buffalo_data_covs$pres_cent) %>% 
  writeRaster(paste0("buffalo_pres_cent", nxn_cells, "x", nxn_cells, "_lag_", hourly_lag, "hr_n", n_samples, ".tif"), overwrite = T)