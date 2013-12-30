(use srfi-1) ; chicken-scheme-specific import

;; MLP

(define (logistic-function x)
  (/ 1 (+ 1 (exp (- x)))))
(define (logistic-function-derivative x)
  (* (logistic-function x) (- 1 (logistic-function x))))
(define (tanh x)
  (cond ((< x -20) -1.0)
        ((> x 20) 1.0)
        (else (let ((y (exp (- (* 2 x)))))
                (/ (- 1 y) (+ 1 y))))))
(define (tanh-derivative x)
  (cond ((or (< x -20)
            (> x 20)) 0)
        (else (- 1 (expt (tanh x) 2)))))


(define (propagate-through-neurons inputs af)
  (map af inputs))

(define (propagate-through-weights neuron-outputs weights)
  (map (lambda (output weights)
         (map (lambda (weight) (* weight output))
              weights))
       neuron-outputs
       weights))

(define (prepare-inputs raw-inputs)
  (fold (lambda (input result)
          (map + input result))
        (make-list (length (car raw-inputs)) 0)
        raw-inputs))

(define (propagate weights inputs af)
  (if (null? weights)
      '()
      (let* ((neuron-outputs (propagate-through-neurons inputs af))
             (next-raw-inputs (propagate-through-weights neuron-outputs (car weights)))
             (next-inputs (prepare-inputs next-raw-inputs)))
        (cons (list inputs neuron-outputs next-raw-inputs next-inputs)
              (propagate (cdr weights) next-inputs af)))))


;; backpropagation
;; http://www.cs.bham.ac.uk/~jxb/NN/l7.pdf

(define (backpropagate-further previous-deltas weights propagation-results)
  (if (null? (cdr weights))
      previous-deltas
      (let* ((propagation-result (car propagation-results))
             (propagation-result-neuron-inputs (car propagation-result))
             (previous-delta (car previous-deltas))
             (weighted-sum-for-delta (prepare-inputs (propagate-through-weights previous-delta
                                                                                (cadr weights))))
             (current-delta (map *
                                 weighted-sum-for-delta
                                 propagation-result-neuron-inputs)))
        (backpropagate-further (cons current-delta previous-deltas)
                               (cdr weights)
                               (cdr propagation-results)))))

(define (calculate-weight-deltas deltas propagation-result learning-rate)
  (map (lambda (layer-deltas layer-propagation-result)
         (let ((layer-neurons-output (cadr layer-propagation-result)))
           (map (lambda (out)
                  (map (lambda (delta)
                         (* learning-rate
                            delta
                            out))
                       layer-deltas))
                layer-neurons-output)))
       deltas
       propagation-result))

(define (backpropagate weights inputs target-outputs af afd learning-rate)
  (let* ((propagation-result (propagate weights inputs af))
         (reverse-propagation-result (reverse propagation-result))
         (last-hidden-layer-propagation-result (car reverse-propagation-result))
         (output-layer-inputs (cadddr last-hidden-layer-propagation-result))
         (output-layer-outputs (propagate-through-neurons output-layer-inputs af))
         (output-layer-delta (map (lambda (targ out in)
                                    (* (- targ out)
                                       (afd in)))
                                  target-outputs
                                  output-layer-outputs
                                  output-layer-inputs))
         (all-deltas (backpropagate-further (list output-layer-delta)
                                            (reverse weights)
                                            reverse-propagation-result))
         (weight-deltas (calculate-weight-deltas all-deltas propagation-result learning-rate))
         (new-weights (map (lambda (weight-layer delta-layer)
                             (map (lambda (w d)
                                    (map + w d))
                                  weight-layer
                                  delta-layer))
                           weights
                           weight-deltas))
         ;; there's virtually no difference for sigmoids if value is like 1000 or 1e1000,
         ;; but we are getting +/-inf.0 and then +/-nan.0 if numbers are too big
         (corrected-weights (map (lambda (weight-layer)
                                   (map (lambda (weight)
                                          (map (lambda (w)
                                                 (cond ((> w 1000) 1000)
                                                       ((< w -1000) -1000)
                                                       (else w)))
                                               weight))
                                        weight-layer))
                                 new-weights)))
    corrected-weights))


;; helper functions

(define (propagate-final weights inputs af)
  (let* ((propagation-result (propagate weights inputs af))
         (reverse-propagation-result (reverse propagation-result))
         (last-hidden-layer-propagation-result (car reverse-propagation-result))
         (output-layer-inputs (cadddr last-hidden-layer-propagation-result))
         (output-layer-outputs (propagate-through-neurons output-layer-inputs af)))
    output-layer-outputs))

(define (train-list weights inputs outputs af afd learning-rate)
  (if (null? inputs)
      weights
      (train-list (backpropagate weights (car inputs) (car outputs) af afd learning-rate)
                  (cdr inputs)
                  (cdr outputs)
                  af
                  afd
                  learning-rate)))

(define (train weights inputs outputs af afd learning-rate n)
  (if (<= n 0)
      weights
      (train (train-list weights inputs outputs af afd learning-rate)
             inputs
             outputs
             af
             afd
             learning-rate
             (- n 1))))

;; test

(define init '(((0.3 0.5 0.8) (0.44 0.3 0.2))
               ((0.7) (0.8) (0.5))))

(define trained-xor (train init
                           '((0 0) (0 1) (1 0) (1 1))
                           '((0) (1) (1) (0))
                           tanh
                           tanh-derivative
                           0.1
                           20000))

(propagate-final trained-xor
                 '(0 0)
                 tanh)
(propagate-final trained-xor
                 '(0 1)
                 tanh)
(propagate-final trained-xor
                 '(1 0)
                 tanh)
(propagate-final trained-xor
                 '(1 1)
                 tanh)
