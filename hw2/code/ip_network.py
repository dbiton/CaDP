from network import *

class IPNeuralNetwork(NeuralNetwork):
    
    def fit(self, training_data, validation_data=None):
        '''
        Override this function to create and destroy workers
        '''
        # 1. Create Workers
		# (Call Worker() with self.mini_batch_size as the batch_size)
        
        workers = []
        for _ in range():
            workers.put(Worker())
        
		# 2. Set jobs
		
        # Call the parent's fit. Notice how create_batches is called inside super.fit().
        super().fit(training_data, validation_data)
        
        # 3. Stop Workers
        
        raise NotImplementedError("To be implemented")
        
        
    
    def create_batches(self, data, labels, batch_size):
        '''
        Override this function to return self.number_of_batches batches created by workers
		Hint: you can either generate (i.e sample randomly from the training data) the image batches here OR in Worker.run()
        '''
        raise NotImplementedError("To be implemented")

    def __init__(self):
        self.write_lock = Lock()
        self.pipe_read, self.pipe_write = Pipe()
        
    def put(self, msg):
        self.write_lock.acquire()
        try:
            self.pipe_write.send(msg)
        finally:
            self.write_lock.release()

    def get(self):
        # blocks until input is given
        return self.pipe_read.recv()
