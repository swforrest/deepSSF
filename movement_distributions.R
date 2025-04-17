
library(terra)
library(ggplot2)

## Create distance and bearing layers for the movement probability
image_dim <- 101
pixel_size <- 25
center <- image_dim %/% 2

# Create matrices of indices
x <- matrix(rep(0:(image_dim - 1), image_dim), nrow = image_dim, byrow = TRUE)
y <- matrix(rep(0:(image_dim - 1), each = image_dim), nrow = image_dim, byrow = TRUE)

# Compute the distance layer
distance_layer <- sqrt((pixel_size * (x - center))^2 + (pixel_size * (y - center))^2)

# Change the center cell to the average distance from the center to all locations within the pixel
distance_layer[center + 1, center + 1] <- 0.3826 * pixel_size

# Compute the bearing layer
bearing_layer <- atan2(center - y, x - center)

# Convert the distance and bearing matrices to raster layers
distance_layer <- rast(distance_layer)
bearing_layer <- rast(bearing_layer)

# Plot the distance and bearing rasters
# png("outputs/movement_distributions/distance_layer.png", width = 1000, height = 750, res = 250)
terra::plot(distance_layer, main = "Distance from centre")
# dev.off()

# png("outputs/movement_distributions/bearing_layer.png", width = 1000, height = 750, res = 250)
terra::plot(bearing_layer, main = "Bearing")
# dev.off()

distance_values <- terra::values(distance_layer)
bearing_values <- terra::values(bearing_layer)

hist(distance_values[distance_values < 1250], breaks = 25, main = "Histogram of Distance Values", xlab = "Distance")
hist(bearing_values[distance_values < 1250], breaks = 25, main = "Histogram of Bearing Values", xlab = "Bearing")

# Create a data frame with the distance and bearing values
movement_data <- data.frame(
  distance = as.numeric(distance_values),
  bearing = as.numeric(bearing_values)
)

# Plot the movement data

ggplot(movement_data, aes(distance)) +
  geom_histogram(binwidth = 50,
                 fill = "orange", colour = "black",
                 alpha = 0.5) +
  # geom_density(fill = "orange", colour = "black",
  #              alpha = 0.5) +
  scale_x_continuous("Distance (m)",
                     limits = c(0,1250),
                     breaks = seq(0, 1250, by = 250)) +
  scale_y_continuous("Number of cells") +
  ggtitle("Number of cells with increasing radius") +
  theme_bw()

# ggsave("outputs/movement_distributions/movement_data_histogram.png", 
#        width=150, height=90, units="mm", dpi = 300)


# Step length distribution



# Gamma distribution parameters
shape = 2
scale = 100

# Create a sequence of x values (1D) for the gamma distribution
x_1D <- seq(1, 1250, by = 0.1)

# Calculate the gamma density values for the 1D distribution

# power 1
step_dist_1D_df <- data.frame(x = x_1D, 
                              y = dgamma(x_1D, shape = shape, scale = scale) / 
                                sum(dgamma(x_1D, shape = shape, scale = scale)))

# divided by distance
step_dist_1D_df <- data.frame(x = x_1D, 
                              y = (dgamma(x_1D, shape = shape, scale = scale) / x_1D)/
                                sum(dgamma(x_1D, shape = shape, scale = scale) / x_1D))

# power 0.5
step_dist_1D_power_0.5_df <- data.frame(x = x_1D, 
                                        y = (dgamma(x_1D, shape = shape, scale = scale)^0.5) / 
                                               sum(dgamma(x_1D, shape = shape, scale = scale)^0.5))

# divided by distance
step_dist_1D_power_0.5_df <- data.frame(x = x_1D, 
                                        y = ((dgamma(x_1D, shape = shape, scale = scale)^2) / x_1D)/
                                          sum((dgamma(x_1D, shape = shape, scale = scale)^2) / x_1D))


step_dist_1D_power_0.25_df <- data.frame(x = x_1D, y = (dgamma(x_1D, shape = shape, scale = scale)^0.25)/sum(dgamma(x_1D, shape = shape, scale = scale)^0.25))

step_dist_1D_power_0.1_df <- data.frame(x = x_1D, y = dgamma(x_1D, shape = shape, scale = scale)^0.1)

# Plot the gamma distribution
ggplot() +
  geom_line(data = step_dist_1D_df, aes(x, y), linetype = "dashed") +
  geom_line(data = step_dist_1D_power_0.5_df, aes(x, y), linetype = "dotdash") +
  # geom_line(data = step_dist_1D_power_0.25_df, aes(x, y)) +
  # geom_line(data = step_dist_1D_power_0.1_df, aes(x, y)) +
  labs(title = "Gamma distribution in one dimension",
       x = "Step length (m)",
       y = "Density") +
  theme_bw()

# ggsave("outputs/movement_distributions/1D_gamma_distribution.png", 
#        width=150, height=90, units="mm", dpi = 300)



# Get the distance values from the distance layer
distance_values <- terra::values(distance_layer)
# Use the distance_layer raster as a template
gamma_step_dist_2D <- distance_layer
# Calculate the gamma density values at the distances from centre
gamma_step_dist_2D[] <- dgamma(distance_values, 
                                shape = shape, 
                                scale = scale) / sum(dgamma(distance_values, 
                                                        shape = shape, 
                                                        scale = scale))
                         
gamma_step_dist_2D[] <- (dgamma(distance_values, 
                                 shape = shape, 
                                 scale = scale) / distance_values) / sum(dgamma(distance_values, 
                                                                                      shape = shape, 
                                                                                      scale = scale) / distance_values)

gamma_step_dist_2D[] <- ((dgamma(distance_values, 
                                 shape = shape, 
                                 scale = scale)^0.5) / distance_values) / sum((dgamma(distance_values, 
                                                                                     shape = shape, 
                                                                                     scale = scale)^0.5) / distance_values)

# Plot the gamma distribution
# png("outputs/movement_distributions/2D_gamma_layer.png", width = 1000, height = 750, res = 250)
plot(gamma_step_dist_2D, main = "2D gamma distribution")
# dev.off()

# Get the values of the two-dimensional gamma distribution
step_dist_2D_values <- terra::values(gamma_step_dist_2D)
# sum(step_dist_2D_values)

# Sample from the two-dimensional gamma distribution
step_dist_2D_samples <- data.frame(
  "steps" = sample(x = distance_values,
                   size = 1e5, 
                   replace = TRUE, 
                   prob = step_dist_2D_values))

# Plot the density of the sampled values against the 1D gamma distribution
ggplot() +
  geom_line(data = step_dist_1D_df, 
            aes(x = x, y = y), colour = "black") +
  geom_density(data = step_dist_2D_samples, 
               aes(x = steps), 
               fill = "orange", colour = "black",
               alpha = 0.5) +
  scale_x_continuous(limits = c(0,1250), breaks = seq(0, 1250, by = 250)) +
  labs(title = "Samples from 2D gamma distribution",
       x = "Step length (m)",
       y = "Density") +
  theme_bw()

# ggsave("outputs/movement_distributions/2D_gamma_samples.png", 
#        width=150, height=90, units="mm", dpi = 300)




# Get the distance values from the distance layer
distance_values <- terra::values(distance_layer)
# Use the distance_layer raster as a template
gamma_step_dist_2D_corr <- distance_layer
# Calculate the gamma density values at the distances from centre
gamma_step_dist_2D_corr[] <- dgamma(distance_values, 
                                    shape = shape, 
                                    scale = scale) / distance_values

# Plot the gamma distribution
png("outputs/movement_distributions/2D_gamma_layer_corrected.png", width = 1000, height = 750, res = 250)
plot(gamma_step_dist_2D_corr, main = "Corrected 2D gamma distribution")
dev.off()

# Get the values of the two-dimensional gamma distribution
gamma_step_dist_2D_corr_values <- terra::values(gamma_step_dist_2D_corr)
# sum(step_dist_2D_values)

# Sample from the two-dimensional gamma distribution
step_dist_2D_corr_samples <- data.frame(
  "steps" = sample(x = distance_values,
                   size = 1e5, 
                   replace = TRUE, 
                   prob = gamma_step_dist_2D_corr_values))

# Plot the density of the sampled values against the 1D gamma distribution
ggplot() +
  geom_line(data = step_dist_1D_df, 
            aes(x = x, y = y), colour = "black") +
  geom_density(data = step_dist_2D_corr_samples, 
               aes(x = steps), 
               fill = "skyblue", colour = "black",
               alpha = 0.5) +
  scale_x_continuous(limits = c(0,1250), breaks = seq(0, 1250, by = 250)) +
  labs(title = "Samples from corrected 2D gamma distribution",
       x = "Step length (m)",
       y = "Density") +
  theme_bw()

ggsave("outputs/movement_distributions/2D_gamma_only_corrected_samples.png", 
       width=150, height=90, units="mm", dpi = 300)


# Sample from the two-dimensional gamma distribution
step_dist_2D_samples <- rbind(
   data.frame(type = "steps", 
              step_lengths = sample(x = distance_values,
                            size = 1e5, 
                            replace = TRUE, 
                            prob = step_dist_2D_values)),
  
  data.frame(type = "corrected_steps",
             step_lengths = sample(x = distance_values,
                                   size = 1e5, 
                                   replace = TRUE, 
                                   prob = step_dist_2D_values / (2*pi*distance_values))))

# Plot the density of the sampled values against the 1D gamma distribution
ggplot() +
  geom_density(data = step_dist_2D_samples, 
               aes(x = step_lengths, fill = type),
               colour = "black",
               alpha = 0.5) +
  geom_line(data = step_dist_1D_df, 
            aes(x = x, y = y), colour = "black") +
  scale_x_continuous(limits = c(0,1250), breaks = seq(0, 1250, by = 250)) +
  scale_fill_manual(values = c("steps" = "orange", "corrected_steps" = "skyblue"),
                    name = "Steps") +
  labs(title = "Samples from 2D gamma distribution",
       x = "Step length (m)",
       y = "Density") +
  theme_bw() +
  theme(legend.position = c(0.85, 0.8))

ggsave("outputs/movement_distributions/2D_gamma_corrected_samples.png", 
       width=150, height=90, units="mm", dpi = 300)


# geom_density(data = step_dist_1D_samples,
#              aes(x = samples), 
#              fill = "blue", colour = "black",
#              alpha = 0.5) +


# Plot the gamma distribution
ggplot(data.frame(x = x_1D, y = dgamma(x_1D, shape = shape, scale = scale)), aes(x = x, y = y)) +
  geom_line() +
  labs(title = "Gamma Distribution",
       x = "x",
       y = "Density") +
  theme_bw()

step_dist_1D_samples <- data.frame(samples = sample(x_1D, size = 1e5, replace = TRUE, prob = (2*pi*x_1D) * dgamma(x_1D, shape = shape, scale = scale)))

ggplot() +
  geom_density(data = step_dist_1D_samples, aes(x = samples)) +
  labs(title = "Gamma Distribution",
       x = "x",
       y = "Density") +
  theme_bw()


step_dist_2D <- distance_layer
step_dist_2D[] <- dgamma(distance_values, 
                         shape = shape, 
                         scale = scale) #/ (2*pi*distance_values)

plot(step_dist_2D)

# Plot the 2D step length distribution
step_dist_2D_values <- terra::values(step_dist_2D)


step_dist_2D_df <- data.frame(distance_values, step_dist_2D_values)


step_dist_2D_samples <- data.frame(sample(x = distance_values,
                                          size = 1e5, 
                                          replace = TRUE, 
                                          prob = step_dist_2D_values))

hist(step_dist_2D_samples, probability = TRUE, breaks = 50)
     



