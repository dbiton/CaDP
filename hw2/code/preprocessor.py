import multiprocessing as mp
import random

class Worker(mp.Process):
    
    def __init__(self, jobs, result, training_data, batch_size):
        super().__init__()

        ''' Initialize Worker and it's members.

        Parameters
        ----------
        jobs: JoinableQueue
            A jobs Queue for the worker.
        result: Queue
            A results Queue for the worker to put it's results in.
		training_data: 
			A tuple of (training images array, image lables array)
		batch_size:
			workers batch size of images (mini batch size)
        
        You should add parameters if you think you need to.
        '''
        self.jobs = jobs
        self.result = result
        self.training_data = training_data
        self.batch_size = batch_size

    @staticmethod
    def rotate(image, angle):
        '''Rotate given image to the given angle

        Parameters
        ----------
        image : numpy array
            An array of size 784 of pixels
        angle : int
            The angle to rotate the image
            
        Return
        ------
        An numpy array of same shape
        '''
        return rotate(image, angle, reshape=False)
    
    @staticmethod
    def shift(image, dx, dy):
        '''Shift given image

        Parameters
        ----------
        image : numpy array
            An array of shape 784 of pixels
        dx : int
            The number of pixels to move in the x-axis
        dy : int
            The number of pixels to move in the y-axis
            
        Return
        ------
        An numpy array of same shape
        '''
        image_shifted = image.roll(x, -dx).roll(y,-dy)
        for x in range(image.shape[0]-dx,image.shape[0]):
            for y in range(image.shape[1]-dy,image.shape[1]):
                image_shifted[x,y]=0
        return image_shifted
    
    @staticmethod
    def step_func(image, steps):
        '''Transform the image pixels acording to the step function

        Parameters
        ----------
        image : numpy array
            An array of shape 784 of pixels
        steps : int
            The number of steps between 0 and 1

        Return
        ------
        An numpy array of same shape
        '''
    return (1/(steps-1)) * np.floor(image*steps)

    @staticmethod
    def skew(image, tilt):
        '''Skew the image

        Parameters
        ----------
        image : numpy array
            An array of size 784 of pixels
        tilt : float
            The skew paramater

        Return
        ------
        An numpy array of same shape
        '''
    image_skewed = np.zeros(image.shape)
    for x in range(0, image.shape[0]):
            for y in range(0,image.shape[1]):
                image_skewed[x,y] = image[y,x+y*tilt]
    return image_skewed

    def process_image(self, image):
        '''Apply the image process functions
		Experiment with the random bounds for the functions to see which produces good accuracies.

        Parameters
        ----------
        image: numpy array
            An array of size 784 of pixels

        Return
        ------
        An numpy array of same shape
        '''
    image_processed = image
    min_iter = 2
    max_iter = 8
    img_width = image.shape[0]
    img_height = image.shape[1]
    iterations = random.randrange(min_iter,max_iter)
    for i in range (iterations):
        operator_idx = random.randrange(0,3)
        #rotate
        if (operator_idx==0):
            angle = random.randrange(359)
            image_processed = rotate(image_processed,angle)
        #shift
        if (operator_idx==1):
            # at most, we'll allow the image to be shifted by half it's width and height
            dx = random.randrange(img_width // (2*img_width))
            dy = random.randrange(img_height // (2*img_height))
            image_processed = shift(image_processed,dx,dy)
        #step
        if (operator_idx==2):
            # at most, we'll allow the image to be stepped on (ha ha) 16 times
            steps = random.randrange(16//iterations)
            image_processed = step_func(image_processed, steps)
        #skew
        if (operator_idx==3):
            # at most, we'll allow the image to be tilted by 1
            tilt = random.randrange(1/iterations)
            image_processed = skew(image_processed, tilt)
    return image_processed
    
    def run(self):
        '''Process images from the jobs queue and add the result to the result queue.
		Hint: you can either generate (i.e sample randomly from the training data)
		the image batches here OR in ip_network.create_batches
        '''
        for _ in range(batch_size):
            job = jobs.get()
            result.put(process_image(job))
            jobs.task_done()
            
        raise NotImplementedError("To be implemented")
