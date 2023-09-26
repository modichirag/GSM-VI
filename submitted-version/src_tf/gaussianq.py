###Implementation of different Gaussian families
###
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors


class MF_Gaussian(tf.Module):
    '''Mean field Gaussian family
    '''
    def __init__(self, d, mu=0, scale=1, name=None, dtype=tf.float32):
        super(MF_Gaussian, self).__init__(name=name)
        self.d = d
        self.loc = tf.Variable(tf.zeros(shape=[self.d]) + mu, name='loc', dtype=dtype)
        self.std = tf.Variable(tf.ones(shape=[self.d]) * scale, name='std', dtype=dtype) 
        self.noise = tfd.MultivariateNormalDiag(loc=tf.zeros(self.d))      
    
    
    @property
    def q(self):
        """Variational distribution"""
        #return tfd.Normal(self.loc, tf.nn.softplus(self.std))
        return tfd.MultivariateNormalDiag(loc=self.loc, scale_diag=tf.nn.softplus(self.std))
        #return tfd.Normal(self.loc, tf.exp(self.std))


    def __call__(self, x):
        return self.log_prob(x)

    def log_prob(self, x):
        return self.q.log_prob(x)

    def sample(self, n=1, sample_shape=None):
        return self.q.sample(n)

    def forward(self, z):
        x = self.loc + z*tf.nn.softplus(self.std)
        #x = self.loc + z*tf.exp(self.std)
        return x

    def inverse(self, x):
        z = (x - self.loc)/tf.nn.softplus(self.std)
        #z = (x - self.loc)/tf.exp(self.std)
        return z




class FR_Gaussian(tf.Module):
    '''Full rank Gaussian family with covariance matrix parametrized in terms of Cholesky factors
    '''
    def __init__(self, d, mu=0., scale=1., name=None, dtype=tf.float32):
        super(FR_Gaussian, self).__init__(name=name)
        self.d = d
        self.loc = tf.Variable(tf.zeros(shape=[self.d], dtype=dtype) + mu, name='loc')
        self.scale = tfp.util.TransformedVariable(
           tf.eye(d, dtype=dtype) *scale, tfp.bijectors.FillScaleTriL(),
           name="rascale_tril")
        # self.scale = tf.cast(tfp.util.TransformedVariable(
        #     tf.eye(d) *scale, tfp.bijectors.FillScaleTriL(),
        #     name="rascale_tril"), dtype=dtype)
        self.noise = tfd.MultivariateNormalDiag(loc=tf.zeros(self.d, dtype=dtype))      
    
    
    @property
    def q(self):
        """Variational distribution"""
        return tfd.MultivariateNormalTriL(loc = self.loc, scale_tril = self.scale)


    def __call__(self, x):
        return self.log_prob(x)

    def log_prob(self, x):
        return self.q.log_prob(x)

    def sample(self, n=1, sample_shape=None):
        return self.q.sample(n)

    def forward(self, z):
        #x = self.loc + self.scale @ z
        #x = self.loc + tf.matmul(self.scale , z)
        x = self.loc + tf.matmul(z, self.scale)
        return x

    def inverse(self, x):
        xm = (x - self.loc)
        #z = tf.linalg.inv(self.scale)@xm
        #z = tf.matmul(tf.linalg.inv(self.scale), xm)
        z = tf.matmul(xm, tf.linalg.inv(self.scale))
        return z



    
class MF_Gaussian_mixture(tf.Module):
    '''Mixture of mean field Gaussians
    '''
    def __init__(self, n, d, mu=0, scale=1, name=None):
        super(MF_Gaussian_mixture, self).__init__(name=name)
        self.n = n
        self.d = d
        
        self.p = tf.Variable(tf.ones(shape=[self.n])/self.n, name='pi')
        self.loc = tf.Variable(tf.zeros(shape=[self.n, self.d]) + mu, name='loc')
        self.std = tf.Variable(tf.ones(shape=[self.n, self.d]) * scale, name='std') 
    
    
    @property
    def q(self):
        """Variational distribution"""
        q = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=tf.nn.softplus(self.p)),
            components_distribution=tfd.MultivariateNormalDiag(
                loc=self.loc,
                scale_diag=tf.nn.softplus(self.std)
            )
        )
        return q

    def __call__(self, x):
        return self.log_prob(x)

    def log_prob(self, x):
        return self.q.log_prob(x)

    def sample(self, n=1, sample_shape=None):
        return self.q.sample(n)

    def forward(self, z):
        #x = self.loc + self.scale @ z
        #x = self.loc + tf.matmul(self.scale , z)
        x = self.loc + tf.matmul(z, self.scale)
        return x

    def inverse(self, x):
        xm = (x - self.loc)
        #z = tf.linalg.inv(self.scale)@xm
        #z = tf.matmul(tf.linalg.inv(self.scale), xm)
        z = tf.matmul(xm, tf.linalg.inv(self.scale))
        return z



class FR_Gaussian_cov(tf.Module):
    '''Full rank Gaussian family with covariance matrix parametrized as full covariance matrix
    '''
    def __init__(self, d, mu=0, scale=1, name=None, dtype=tf.float32):
        super(FR_Gaussian_cov, self).__init__(name=name)
        self.d = d
        self.loc = tf.Variable(tf.zeros(shape=[self.d], dtype=dtype) + mu, name='loc')
        self.cov = tf.Variable(tf.constant(np.identity(d), dtype=dtype) * scale**2, name='cov') #note scale**2
        self.noise = tfd.MultivariateNormalDiag(loc=tf.zeros(self.d))

    @property
    def q(self):
        """Variational distribution"""
        return tfd.MultivariateNormalFullCovariance(loc = self.loc, covariance_matrix= self.cov)
    
    def __call__(self, x):
        return self.log_prob(x)

    def log_prob(self, x):
        return self.q.log_prob(x)
    
    def log_likelihood(self, x):
        return self.q.log_prob(x)
    
    @tf.function
    def grad_log_likelihood(self, q):
        with tf.GradientTape() as tape:
            tape.watch(q)
            lp = self.log_prob(q)
        grad = tape.gradient(lp, q)
        return grad

    def sample(self, n=1, sample_shape=None):
        return self.q.sample(n)

    def forward(self, z):
        x = self.loc + tf.matmul(z, self.scale)
        return x

    def inverse(self, x):
        xm = (x - self.loc)
        z = tf.matmul(xm, tf.linalg.inv(self.scale))
        return z



class SinhArcsinhTransformation(tf.Module):
    '''SinhArchSinh transformation of a full rank Gaussian family
    '''
    def __init__(self, d, loc=0., scale=1., skewness=0., tailweight=1., distribution=None, name=None, dtype=tf.float32):
        super(SinhArcsinhTransformation, self).__init__(name=name)
        self.d = d
        self.loc = tf.Variable(tf.constant(loc, dtype=dtype), name='loc')
        #self.scale = tf.Variable(tf.constant(np.identity(d), dtype=dtype) * scale, name='scale')
        self.scale = tf.Variable(tf.constant(scale, dtype=dtype), name='scale')
        self.skewness = skewness
        self.tailweight = tailweight
        self.distribution = distribution
        

    @property
    def q(self):
        """Variational distribution"""
        return tfd.SinhArcsinh(self.loc, self.scale, skewness=self.skewness,
                               tailweight=self.tailweight, distribution=self.distribution)
    
    def __call__(self, x):
        return self.log_prob(x)

    def log_prob(self, x):
        return self.q.log_prob(x)
    
    def log_likelihood(self, x):
        return self.q.log_prob(x)
    
    @tf.function
    def grad_log_likelihood(self, q):
        with tf.GradientTape() as tape:
            tape.watch(q)
            lp = self.log_prob(q)
        grad = tape.gradient(lp, q)
        return grad

    def sample(self, n=1, sample_shape=None):
        return self.q.sample(n)

    def forward(self, z):
        x = self.loc + tf.matmul(z, self.scale)
        return x

    def inverse(self, x):
        xm = (x - self.loc)
        z = tf.matmul(xm, tf.linalg.inv(self.scale))
        return z
    
