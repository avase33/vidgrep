package pipeline

import "testing"

func TestQueuePushPop(t *testing.T) {
	q := NewQueue(2)
	if !q.Push(Job{VideoID: "a"}) {
		t.Fatal("push should succeed")
	}
	j := <-q.Chan()
	if j.VideoID != "a" {
		t.Fatalf("got %q", j.VideoID)
	}
}

func TestQueueFullReturnsFalse(t *testing.T) {
	q := NewQueue(1)
	q.Push(Job{VideoID: "a"})
	if q.Push(Job{VideoID: "b"}) {
		t.Fatal("second push should fail on a full queue")
	}
}

func TestStoreUpdateNotifiesSubscribers(t *testing.T) {
	s := NewStore()
	s.Create("v")
	sub := s.Subscribe("v")

	s.update("v", func(st *Status) {
		st.Frames = 5
		st.Done = true
	})

	got := <-sub
	if got.Frames != 5 || !got.Done {
		t.Fatalf("subscriber got %+v", got)
	}
	snap, ok := s.Get("v")
	if !ok || snap.Frames != 5 {
		t.Fatalf("store snapshot %+v ok=%v", snap, ok)
	}
}
